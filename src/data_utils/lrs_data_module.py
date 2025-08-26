import json
import torch
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
from itertools import chain
from collections import defaultdict
from typing import List, Optional
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, random_split

import random
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)



@dataclass
class AVSRItem:
    speech_token: Optional[torch.Tensor]
    visual_feature: torch.Tensor
    text_token: Optional[torch.Tensor]
    key: Optional[str]


@dataclass
class AudioVisualBatch:
    keys: Optional[List[str]]
    # individual tensor and information
    speech_token: Optional[torch.Tensor]
    speech_token_len: Optional[torch.Tensor]
    speech_mask: torch.Tensor

    visual_feature: torch.Tensor    
    visual_feature_len: torch.Tensor
    visual_mask: torch.Tensor

    text_token: Optional[torch.Tensor]
    text_token_len: Optional[torch.Tensor]
    text_mask: torch.Tensor
    # prompt and prompt length
    prompt_token: torch.Tensor
    prompt_token_len: torch.Tensor
    

@dataclass
class AVQFormerBatch:
    keys: Optional[List[str]]

    visual_feature: torch.Tensor    
    visual_feature_len: torch.Tensor
    visual_mask: torch.Tensor

    query_mask: torch.Tensor
    speech_mask: torch.Tensor

    text_token: Optional[torch.Tensor]
    text_token_len: Optional[torch.Tensor]
    text_mask: torch.Tensor
    # prompt and prompt length
    prompt_token: torch.Tensor
    prompt_token_len: torch.Tensor
    


class AudioVisualDataset(Dataset):
    def __init__(self, data_list, min_len=0, max_len=9999999, tokenizer=None, is_train=False, fusion_mode="x-attn"):
        assert tokenizer is not None, "SPIRITLM tokenizer must be provided"
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.data = []
        data_len = []
 
        num_filtered = 0
        for i, item in enumerate(data_list):
            speech_len = item['unit_len']
            text_len = item['token_len']
            total_len = speech_len + text_len
            if 'visual_path' not in item:
                # Fallback for evaluation data compatibility
                visual_path = os.environ.get('DEFAULT_VISUAL_PATH', './data/default_visual_feat.npy')
            else:
                visual_path = item['visual_path']
                visual_path = visual_path.replace('/video/', '/face_video/')
            if not os.path.exists(visual_path):
                num_filtered += 1
                continue
            item['visual_path'] = visual_path
            if is_train and speech_len + text_len > 512:
                num_filtered += 1
                continue
            if speech_len > 20 * text_len:
                num_filtered += 1
                continue

            # speech and text token determine the sample prompt length
            data_len.append(total_len)
            self.data.append(item)
        self.data_len = data_len
        logging.info(f"Filtered {num_filtered} samples due to length constraints.")

    def __len__(self):
        return len(self.data)
    
    def lengths(self):
        return self.data_len

    

    def __getitem__(self, idx):
        item = self.data[idx]
        key = item['key']
        speech_token_str = item['base_str'] # for avsr, we don't use style tokens
        speech_token = self.tokenizer(speech_token_str, return_tensors="pt")['input_ids'].squeeze(0)
        # decode = self.tokenizer.decode(speech_token[0]) # debug purpose
        text_token = torch.tensor(item['tokens'])
        visual_feat = np.load(item['visual_path'])
        visual_feature = torch.from_numpy(visual_feat) # TxD
        # resample visual feature to match speech token length
        if self.fusion_mode == "concate" or self.fusion_mode == "x-attn":
            # visual_feature: (T, feature_dim) -> shape (1, feature_dim, T) 
            visual_feature = visual_feature.unsqueeze(0).transpose(1, 2)
            # Resample to match speech token length
            visual_feature = F.interpolate(visual_feature, size=speech_token.shape[0], mode='linear', align_corners=False)
            # Revert back to (T, feature_dim)
            visual_feature = visual_feature.transpose(1, 2).squeeze(0)
            assert visual_feature.shape[0] == speech_token.shape[0], f"Visual feature shape {visual_feature.shape} does not match speech token shape {speech_token.shape}"

        return AVSRItem(
            speech_token=speech_token,
            visual_feature=visual_feature,
            text_token=text_token,
            key=key
        )

class SpeechSampler(Sampler):
    def __init__(self, batch_size: int, lengths: Optional[List[int]] = None, generator=None):
        super().__init__(None)
        self.batch_size = batch_size
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.tolist()
        self.lengths = lengths
        self.generator = generator

    @staticmethod
    def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
        if mega_batch_mult is None:
            mega_batch_mult = min(len(lengths) // (batch_size * 4), 512)
            if mega_batch_mult == 0:
                mega_batch_mult = 1

        indices = torch.randperm(len(lengths), generator=generator)
        megabatch_size = mega_batch_mult * batch_size
        megabatches = [
            indices[i: i + megabatch_size].tolist()
            for i in range(0, len(lengths), megabatch_size)
        ]
        megabatches = [
            sorted(megabatch, key=lambda i: lengths[i], reverse=True)
            for megabatch in megabatches
        ]

        megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
        max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
        megabatches[0][0], megabatches[max_idx][0] = (
            megabatches[max_idx][0],
            megabatches[0][0],
        )

        return [i for megabatch in megabatches for i in megabatch]

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = self.get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)


class AudioVisualDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, batch_size=32, num_workers=4, seed=42, drop_modality=False, drop_audio_ratio=0.5, fusion_mode="x-attn"):
        super().__init__()
        self.tokenizer = tokenizer
        # for AVSR task, we have several hard code tokens for prompt construction
        self.tokenizer_vocab = tokenizer.get_vocab()
        # AV-> Text, we have <bos> please transcribe this audio visual clip <start_of_av> placeholders <start of text> text tokens <eos>
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.start_of_av_token = self.tokenizer_vocab['[Madeuptoken32766]']
        # for A->T, we have another special token to indicate that visual modality is not fused into it
        self.start_of_audio_token = self.tokenizer_vocab['[Madeuptoken32765]']
        # for V->, we have another special token to indicate that audio modality is not fused into it
        self.start_of_visual_token = self.tokenizer_vocab['[Madeuptoken32764]']
        self.start_of_text_token = self.tokenizer_vocab['[Madeuptoken32763]']
        self.speech_mask_token = self.tokenizer_vocab['[Madeuptoken32762]']
        logging.info(f"<start_of_av>: {self.start_of_av_token}")
        logging.info(f"<start_of_audio>: {self.start_of_audio_token}")
        logging.info(f"<start_of_visual>: {self.start_of_visual_token}")
        logging.info(f"<start_of_text>: {self.start_of_text_token}")

        self.train_json = os.path.join(data_dir, "train.jsonl")
        self.valid_json = os.path.join(data_dir, "valid.jsonl")
        self.test_json = os.path.join(data_dir, "test_snr_0.jsonl")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.drop_modality = drop_modality
        if drop_modality:
            self.drop_audio_ratio = drop_audio_ratio
        else:
            self.drop_audio_ratio = 0.0
        self.fusion_mode = fusion_mode

    def load_and_split_data(self, json_path, shuffle=True):
        """Loads JSONL data and performs an 80:20 split."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        if shuffle:
            random.seed(self.seed)  # Ensure reproducibility
            random.shuffle(data)  # Shuffle the dataset
        return data
    
    def setup(self, stage=None):
        train_data = self.load_and_split_data(self.train_json)
        val_data = self.load_and_split_data(self.valid_json, shuffle=False)
        test_data = self.load_and_split_data(self.test_json, shuffle=False)

        self.train_dataset = AudioVisualDataset(train_data, min_len=3, max_len=128, tokenizer=self.tokenizer, is_train=True, fusion_mode=self.fusion_mode)
        self.val_dataset = AudioVisualDataset(val_data, max_len=128, tokenizer=self.tokenizer, fusion_mode=self.fusion_mode)
        self.test_dataset = AudioVisualDataset(test_data, max_len=128, tokenizer=self.tokenizer, fusion_mode=self.fusion_mode)
        # Log dataset statistics
        logging.info(f"Train dataset size: {len(self.train_dataset)}")
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")



    def train_dataloader(self):
        lengths = self.train_dataset.lengths()
        if self.fusion_mode == "qformer":
            train_collate_fn = lambda batch: self.qformer_collate_fn(batch)
        else:
            train_collate_fn = lambda batch: self.collate_fn(batch, mask_percentage=0.5)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=SpeechSampler(self.batch_size, lengths),
            num_workers=self.num_workers,
            collate_fn=train_collate_fn
        )

    def val_dataloader(self):
        lengths = self.val_dataset.lengths()
        # Create a validation-specific collate function with drop_ratio=0
        if self.fusion_mode == "qformer": # let validation set be all corrupted so that it's assessing model's robustness to use visual modality
            val_collate_fn = lambda batch: self.qformer_collate_fn(batch, override_drop_ratio=1.0, mask_percentage=0.5)
        else:
            val_collate_fn = lambda batch: self.collate_fn(batch, override_drop_ratio=0.0, mask_percentage=0.0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=SpeechSampler(self.batch_size, lengths),
            num_workers=self.num_workers,
            collate_fn=val_collate_fn
        )

    def test_dataloader(self):
        if self.fusion_mode == "qformer":
            test_collate_fn = lambda batch: self.qformer_collate_fn(batch, override_drop_ratio=1, mask_percentage=1)
        else:
            test_collate_fn = lambda batch: self.collate_fn(batch, override_drop_ratio=1, mask_percentage=0.7)

        return DataLoader(
            self.test_dataset,
            batch_size=16, # for simplicity
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=test_collate_fn
        )
    

    def collate_fn(self, batch: List[AVSRItem], override_drop_ratio=None, mask_percentage=None):
        # Use override_drop_ratio if provided, otherwise use class default
        drop_ratio = override_drop_ratio if override_drop_ratio is not None else self.drop_audio_ratio
        drop_audio = torch.rand(len(batch)) < drop_ratio


        # use audio visual modalities but randomly replace some of the speech tokens with [mask_token]
        speech_tokens = [item.speech_token for item in batch]
        speech_lens = torch.tensor([st.shape[0] for st in speech_tokens])
        visual_features = [item.visual_feature for item in batch]
        visual_lens = torch.tensor([vf.shape[0] for vf in visual_features])
        text_tokens = [item.text_token for item in batch]
        text_lens = torch.tensor([tt.shape[0] for tt in text_tokens])

        # Pad visual features
        visual_feature_max_len = visual_lens.max()
        visual_feature_dim = visual_features[0].shape[1]
        visual_features_padded = torch.zeros(len(batch), visual_feature_max_len, visual_feature_dim)
        visual_features_mask = torch.zeros(len(batch), visual_feature_max_len, dtype=torch.bool)
        for i, vf in enumerate(visual_features):
            visual_features_padded[i, :vf.shape[0], :] = vf  # left-align padding
            visual_features_mask[i, :vf.shape[0]] = True

       

        # craft av only prompt and pad the tokens
        speech_text_lens = speech_lens + text_lens
        speech_text_lens_max = speech_text_lens.max()

        for i, speech_token in enumerate(speech_tokens):
            if drop_audio[i]:
                # Find speech token positions
                speech_len = speech_token.shape[0]
                
                # Random masking percentage between 30% and 70%
                if mask_percentage is None:
                    mask_percentage = torch.rand(1).item() * 0.55 + 0.15 # random in [0.15, 0.7]
                num_tokens_to_mask = int(speech_len * mask_percentage)
                
                # Randomly select positions to mask
                mask_positions = torch.randperm(speech_len)[:num_tokens_to_mask]
                # Apply masking
                speech_tokens[i][mask_positions] = self.speech_mask_token

        speech_len_max = speech_lens.max()
        speech_tokens_padded = torch.full((len(batch), speech_len_max), self.pad_token_id, dtype=torch.long)
        for i, st in enumerate(speech_tokens):
            speech_tokens_padded[i, :st.shape[0]] = st



        # Prompt: <bos> Transcribe the following audio visual clip <start_of_av_token> speech_tokens <start_of_text_token> text_tokens <eos>
        # Calculate max prompt length
        prompt_instruction = "Transcribe the following audio visual clip:\n\n"
        prompt_token = torch.tensor(self.tokenizer.encode(prompt_instruction, add_special_tokens=False))
        prompt_len = prompt_token.shape[0]

        sample_max_len = 1 + prompt_len + 1 + speech_text_lens_max + 1 + 1  # bos + prompt + <start_av> +  speech + <start_text> + text + eos
        
        # Initialize padded prompt tensor
        prompts_padded = torch.full((len(batch), sample_max_len), self.pad_token_id, dtype=torch.long)
        speech_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        text_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        # Build prompts for each item in batch
        for i, (speech_token, text_token) in enumerate(zip(speech_tokens, text_tokens)):
            curr_idx = 0
            cur_is_drop_audio = drop_audio[i]
            
            # Add bos token
            prompts_padded[i, curr_idx] = self.bos_token_id
            curr_idx += 1
            
            # Add prompt instruction tokens
            prompts_padded[i, curr_idx:curr_idx+prompt_len] = prompt_token
            curr_idx += prompt_len
            
            # Add start_of_av token
            prompts_padded[i, curr_idx] = self.start_of_av_token
            curr_idx += 1
            
            # Add speech tokens and mark speech mask
            speech_len = speech_token.shape[0]
            speech_start = curr_idx
            speech_end = curr_idx + speech_len
            prompts_padded[i, speech_start:speech_end] = speech_token
            speech_mask[i, speech_start:speech_end] = True
            curr_idx += speech_len
            
            # Add start_of_text token
            prompts_padded[i, curr_idx] = self.start_of_text_token
            curr_idx += 1
            
            # Add text tokens and mark text mask
            text_len = text_token.shape[0]
            text_start = curr_idx
            text_end = curr_idx + text_len
            prompts_padded[i, text_start:text_end] = text_token
            text_mask[i, text_start:text_end + 1] = True # + 1 to include <eos>
            curr_idx += text_len
            
            # Add eos token
            prompts_padded[i, curr_idx] = self.eos_token_id
        
        # Calculate actual lengths of prompts
        prompt_lens = torch.tensor([1 + prompt_len + 1 + sl + 1 + tl + 1 for sl, tl in zip(speech_lens, text_lens)])

        if speech_mask.sum() == 0:
            logging.error(f"Invalid speech_mask: {speech_mask}, speech_lens: {speech_lens}")
            raise ValueError("speech_mask is all 0")


        return AudioVisualBatch(
            keys=[item.key for item in batch],
            speech_token=speech_tokens_padded,
            speech_token_len=speech_lens,
            speech_mask=speech_mask,
            visual_feature=visual_features_padded,
            visual_feature_len=visual_lens,
            visual_mask=visual_features_mask,
            text_token=text_tokens,
            text_token_len=text_lens,
            text_mask=text_mask,
            prompt_token=prompts_padded,
            prompt_token_len=prompt_lens
        )



    def qformer_collate_fn(self, batch: List[AVSRItem], override_drop_ratio=None, mask_percentage=None):
        # Use override_drop_ratio if provided, otherwise use class default
        drop_ratio = override_drop_ratio if override_drop_ratio is not None else self.drop_audio_ratio
        drop_audio = torch.rand(len(batch)) < drop_ratio


        # use audio visual modalities but randomly replace some of the speech tokens with [mask_token]
        speech_tokens = [item.speech_token for item in batch]
        speech_lens = torch.tensor([st.shape[0] for st in speech_tokens])
        visual_features = [item.visual_feature for item in batch]
        visual_lens = torch.tensor([vf.shape[0] for vf in visual_features])
        text_tokens = [item.text_token for item in batch]
        text_lens = torch.tensor([tt.shape[0] for tt in text_tokens])

        # Pad visual features
        visual_feature_max_len = visual_lens.max()
        visual_feature_dim = visual_features[0].shape[1]
        visual_features_padded = torch.zeros(len(batch), visual_feature_max_len, visual_feature_dim)
        visual_features_mask = torch.zeros(len(batch), visual_feature_max_len, dtype=torch.bool)
        for i, vf in enumerate(visual_features):
            visual_features_padded[i, :vf.shape[0], :] = vf  # left-align padding
            visual_features_mask[i, :vf.shape[0]] = True

        # Create dummy visual tokens with length = visual_feature_length / 5
        query_token_lens = (visual_lens / 5).ceil().to(torch.long)
        # query token les cannot be smaller than 1 but no longer than 128 (max_query_size)
        query_token_lens = torch.max(query_token_lens, torch.ones_like(query_token_lens))
        query_token_lens = torch.min(query_token_lens, torch.ones_like(query_token_lens) * 128)
        query_speech_text_lens = query_token_lens + speech_lens + text_lens
        query_speech_text_lens_max = query_speech_text_lens.max()

        for i, speech_token in enumerate(speech_tokens):
            if drop_audio[i]:
                # Find speech token positions
                speech_len = speech_token.shape[0]
                # Random masking percentage between 30% and 70%
                if mask_percentage is None:
                    mask_percentage = torch.rand(1).item() * 0.6 + 0.2 # random in [0.2, 0.8]
                num_tokens_to_mask = int(speech_len * mask_percentage)
                
                # Randomly select positions to mask
                mask_positions = torch.randperm(speech_len)[:num_tokens_to_mask]
                # Apply masking
                speech_tokens[i][mask_positions] = self.speech_mask_token


        # Prompt: <bos> Transcribe the following audio visual clip <start_of_av_token> speech_tokens <start_of_text_token> text_tokens <eos>
        # Calculate max prompt length
        prompt_instruction = "Transcribe the following audio visual clip:\n\n"
        prompt_token = torch.tensor(self.tokenizer.encode(prompt_instruction, add_special_tokens=False))
        prompt_len = prompt_token.shape[0]

        # bos + prompt + <start_v> visual(query_tokens) + <start_a> +  speech + <start_text> + text + eos
        sample_max_len = 5 + prompt_len + query_speech_text_lens_max # 5 special tokens: bos, prompt, <start_v>, <start_a>, <start_text>
        
        # Initialize padded prompt tensor
        prompts_padded = torch.full((len(batch), sample_max_len), self.pad_token_id, dtype=torch.long)
        speech_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        text_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        query_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        # Build prompts for each item in batch
        prompt_lens = []
        for i, (speech_token, text_token) in enumerate(zip(speech_tokens, text_tokens)):
            curr_idx = 0 
            # Add bos token
            prompts_padded[i, curr_idx] = self.bos_token_id
            curr_idx += 1
            
            # Add prompt instruction tokens
            prompts_padded[i, curr_idx:curr_idx+prompt_len] = prompt_token
            curr_idx += prompt_len
            
            # Add start_of_av token
            prompts_padded[i, curr_idx] = self.start_of_visual_token
            curr_idx += 1

            prompts_padded[i, curr_idx:curr_idx+query_token_lens[i]] = self.pad_token_id
            query_mask[i, curr_idx:curr_idx+query_token_lens[i]] = True
            curr_idx += query_token_lens[i]

            prompts_padded[i, curr_idx] = self.start_of_audio_token
            curr_idx += 1

            
            # Add speech tokens and mark speech mask
            speech_len = speech_token.shape[0]
            speech_start = curr_idx
            speech_end = curr_idx + speech_len
            prompts_padded[i, speech_start:speech_end] = speech_token
            speech_mask[i, speech_start:speech_end] = True
            curr_idx += speech_len
            
            # Add start_of_text token
            prompts_padded[i, curr_idx] = self.start_of_text_token
            curr_idx += 1
            
            # Add text tokens and mark text mask
            text_len = text_token.shape[0]
            text_start = curr_idx
            text_end = curr_idx + text_len
            prompts_padded[i, text_start:text_end] = text_token
            text_mask[i, text_start:text_end + 1] = True # + 1 to include <eos>
            curr_idx += text_len
            
            # Add eos token
            prompts_padded[i, curr_idx] = self.eos_token_id
            prompt_lens.append(curr_idx + 1)

        # Calculate actual lengths of prompts
        prompt_lens = torch.tensor(prompt_lens)


        return AVQFormerBatch(
            keys=[item.key for item in batch],
            query_mask=query_mask,
            speech_mask=speech_mask,
            visual_feature=visual_features_padded,
            visual_feature_len=visual_lens,
            visual_mask=visual_features_mask,
            text_token=text_tokens,
            text_token_len=text_lens,
            text_mask=text_mask,
            prompt_token=prompts_padded,
            prompt_token_len=prompt_lens
        )

# test the data module
if __name__ == "__main__":
    data_dir = os.environ.get('DATA_DIR', './data/IEMOCAP/processed/cosyvoice')
    data_module = AudioVisualDataModule(data_dir, num_workers=1, batch_size=4)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        logging.info(f"Visual feature shape: {batch.visual_feature.shape}")
        logging.info(f"Speech feature shape: {batch.speech_feature.shape}")
        break