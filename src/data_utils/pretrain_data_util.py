# different lrs_data_module/mm_data_module for AVSR finetuning, this dataloader is to prepare next-token-prediction data
# for AVLM pretraining. The target objective is the speech units

import json
import torch
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from dataclasses import dataclass
from src.data_utils.lrs_data_module import SpeechSampler
import random
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)



@dataclass
class AVSRItem:
    speech_token: Optional[torch.Tensor]
    jaw_feat: torch.Tensor
    expression_feat: torch.Tensor
    text_token: Optional[torch.Tensor]
    key: Optional[str]


@dataclass
class AudioVisualBatch:
    keys: Optional[List[str]]
    # individual tensor and information
    jaw_feature: torch.Tensor
    expression_feature: torch.Tensor
    visual_feature_len: torch.Tensor
    visual_mask: torch.Tensor

    input_token: torch.Tensor
    input_mask: torch.Tensor
    output_token: torch.Tensor
    labels: torch.Tensor

@dataclass
class AVQFormerBatch:
    keys: Optional[List[str]]
    jaw_feature: torch.Tensor
    expression_feature: torch.Tensor
    visual_feature_len: torch.Tensor
    visual_mask: torch.Tensor

    query_mask: torch.Tensor
    input_token: torch.Tensor
    input_mask: torch.Tensor
    output_token: torch.Tensor
    labels: torch.Tensor




class AudioVisualDataset(Dataset):
    def __init__(self, data_list, min_len=0, max_len=9999999, tokenizer=None, is_train=False, fusion_mode="x-attn", expressive_mode=False):
        assert tokenizer is not None, "SPIRITLM tokenizer must be provided"
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.expressive_mode = expressive_mode
        self.data = []
        data_len = []
        speech_len_limit = 512 if not expressive_mode else 512
 
        num_filtered = 0
        for i, item in enumerate(data_list):
            if expressive_mode:
                speech_len = len(item['expressive_str'].split('['))
            else:   
                speech_len = item['unit_len']
            if fusion_mode == "concate":
                speech_len = len(item['hubert'].split(' '))
            text_len = item['token_len']
            total_len = speech_len + text_len

            if is_train and speech_len > speech_len_limit:
                num_filtered += 1
                continue
            
                
            if speech_len > 20 * text_len:
                num_filtered += 1
                continue

            # speech and text token determine the sample prompt length
            data_len.append(total_len)
            self.data.append(item)
        self.data_len = data_len
        print(f"Filtered {num_filtered} samples due to length constraints.")

    def __len__(self):
        return len(self.data)
    
    def lengths(self):
        return self.data_len

    @staticmethod
    def _feat_resample(feat, length):
        # feat: (T, D), length: int
        feat = feat.unsqueeze(0).transpose(1, 2) # (1, D, T)
        feat = F.interpolate(feat, size=length, mode='linear', align_corners=False).squeeze(0) # (D, T')
        return feat.transpose(0, 1) # (T, D)

    def __getitem__(self, idx):
        item = self.data[idx]
        key = item['key']
        if self.expressive_mode:
            speech_token_str = item['expressive_str']
        else:
            speech_token_str = item['base_str'] # for avsr, we don't use style tokens
        speech_token = self.tokenizer(speech_token_str, return_tensors="pt")['input_ids'].squeeze(0)
        # decode = self.tokenizer.decode(speech_token[0]) # debug purpose
        text_token = torch.tensor(item['tokens'])
        smirk_feat = np.load(item['smirk_path'], allow_pickle=True).item()
        # only use jaw and expression features
        jaw_feat = torch.from_numpy(smirk_feat['jaw_params']) # T x 3
        expression_feat = torch.from_numpy(smirk_feat['expression_params']) # T x 50
 

        # resample visual feature to match speech token length
        if self.fusion_mode == "concate":
            duplicate_speech_token = list(map(lambda x: int(x), item['hubert'].split(' ')))
            duplicate_speech_str = ''.join([f'[Hu{token}]' for token in duplicate_speech_token])
            speech_token = self.tokenizer(duplicate_speech_str, return_tensors="pt")['input_ids'].squeeze(0)

            # visual_feature: (T, feature_dim) -> shape (1, feature_dim, T)
            if jaw_feat.shape[0] != speech_token.shape[0]:
                jaw_feat = self._feat_resample(jaw_feat, speech_token.shape[0])
                expression_feat = self._feat_resample(expression_feat, speech_token.shape[0])
    
        return AVSRItem(
            speech_token=speech_token,
            jaw_feat=jaw_feat,
            expression_feat=expression_feat,
            text_token=text_token,
            key=key)
    



class AVLMPretrainDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, batch_size=32, num_workers=4, seed=42, drop_audio_ratio=0.0, fusion_mode="x-attn", test_drop_ratio=0.0, expressive_mode=False):
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
        self.test_json = os.path.join(data_dir, "test.jsonl")
        # self.test_json = os.path.join(data_dir, "test_snr_10.jsonl")
        # self.test_json = os.path.join(data_dir, "test_snr_0.jsonl")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.drop_audio_ratio = drop_audio_ratio
        self.fusion_mode = fusion_mode
        self.test_drop_ratio = test_drop_ratio
        self.expressive_mode = expressive_mode

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

        self.train_dataset = AudioVisualDataset(train_data, min_len=3, max_len=128, tokenizer=self.tokenizer, is_train=True, fusion_mode=self.fusion_mode, expressive_mode=self.expressive_mode)
        self.val_dataset = AudioVisualDataset(val_data, max_len=128, tokenizer=self.tokenizer, fusion_mode=self.fusion_mode, expressive_mode=self.expressive_mode)
        self.test_dataset = AudioVisualDataset(test_data, max_len=128, tokenizer=self.tokenizer, fusion_mode=self.fusion_mode, expressive_mode=self.expressive_mode)
        # print data stats
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")



    def train_dataloader(self):
        lengths = self.train_dataset.lengths()
        if self.fusion_mode == "qformer":
            train_collate_fn = lambda batch: self.qformer_collate_fn(batch)
        elif self.fusion_mode == "asr_only" or self.fusion_mode == "qformer_infill" or self.fusion_mode == "concate":
            train_collate_fn = lambda batch: self.collate_fn(batch)
        elif self.fusion_mode == "qformer_avsr":
            train_collate_fn = lambda batch: self.qformer_avsr_collate_fn(batch)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=SpeechSampler(self.batch_size, lengths),
            num_workers=self.num_workers,
            # num_workers=1,
            collate_fn=train_collate_fn
        )

    def val_dataloader(self):
        lengths = self.val_dataset.lengths()
        if self.fusion_mode == "qformer":
            val_collate_fn = lambda batch: self.qformer_collate_fn(batch, override_drop_ratio=0.3)
        elif self.fusion_mode == "asr_only" or self.fusion_mode == "qformer_infill":
            val_collate_fn = lambda batch: self.collate_fn(batch, override_drop_ratio=0.3)
        elif self.fusion_mode == "concate":
            val_collate_fn = lambda batch: self.collate_fn(batch, override_drop_ratio=0.0)
        elif self.fusion_mode == "qformer_avsr":
            val_collate_fn = lambda batch: self.qformer_avsr_collate_fn(batch, override_drop_ratio=0.3)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=SpeechSampler(self.batch_size, lengths),
            num_workers=self.num_workers,
            collate_fn=val_collate_fn
        )

    def test_dataloader(self):
        if self.fusion_mode == "qformer":
            test_collate_fn = lambda batch: self.qformer_collate_fn(batch, override_drop_ratio=self.test_drop_ratio)
        elif self.fusion_mode == "asr_only" or self.fusion_mode == "qformer_infill" or self.fusion_mode == "concate":
            test_collate_fn = lambda batch: self.collate_fn(batch, override_drop_ratio=self.test_drop_ratio)
        elif self.fusion_mode == "qformer_avsr":
            test_collate_fn = lambda batch: self.qformer_avsr_collate_fn(batch, override_drop_ratio=self.test_drop_ratio)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")

        return DataLoader(
            self.test_dataset,
            batch_size=16, # for simplicity
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=test_collate_fn
        )
    

    def collate_fn(self, batch: List[AVSRItem], override_drop_ratio=None):
        ####### Prepare visual features #######
        jaw_feats = [item.jaw_feat for item in batch]
        expression_feats = [item.expression_feat for item in batch]
        visual_lens = torch.tensor([jf.shape[0] for jf in jaw_feats])
        visual_feature_max_len = visual_lens.max()
        visual_features_mask = torch.zeros(len(batch), visual_feature_max_len, dtype=torch.bool)
        # Pad visual features
        jaw_features_padded = torch.zeros(len(batch), visual_feature_max_len, jaw_feats[0].shape[1])
        expression_features_padded = torch.zeros(len(batch), visual_feature_max_len, expression_feats[0].shape[1])
        for i, (jf, ef) in enumerate(zip(jaw_feats, expression_feats)):
            jaw_features_padded[i, :jf.shape[0], :] = jf
            expression_features_padded[i, :ef.shape[0], :] = ef
            visual_features_mask[i, :jf.shape[0]] = True

        ####### Prepare input and output sequneces. Input is <bos> speech tokens, and output is speech tokens ####

        speech_tokens = [item.speech_token for item in batch]
        speech_lens = torch.tensor([st.shape[0] for st in speech_tokens])
        speech_len_max = speech_lens.max()
        output_tokens = torch.full((len(batch), speech_len_max), self.pad_token_id, dtype=torch.long)
        input_tokens = torch.full((len(batch), speech_len_max), self.pad_token_id, dtype=torch.long)
        input_tokens[:, 0] = self.bos_token_id
        input_mask = torch.zeros((len(batch), speech_len_max), dtype=torch.bool)
        for i, speech_token in enumerate(speech_tokens):
            # Find speech token positions
            speech_len = speech_token.shape[0]
            if override_drop_ratio is not None:
                num_tokens_to_mask = int(speech_len * override_drop_ratio)
            else:
                num_tokens_to_mask = int(speech_len * self.drop_audio_ratio)
            if num_tokens_to_mask > 0:
                # Randomly select positions to mask
                mask_positions = torch.randperm(speech_len)[:num_tokens_to_mask]
                noise_token = speech_token.clone()
                noise_token[mask_positions] = self.speech_mask_token
            else:
                noise_token = speech_token

            input_tokens[i, 1:speech_len] = noise_token[:-1]
            output_tokens[i, :speech_len] = speech_token
            input_mask[i, :speech_len] = True
        labels = output_tokens.clone()
        labels[output_tokens == self.pad_token_id] = -100


        return AudioVisualBatch(
            keys=[item.key for item in batch],
            input_token=input_tokens,
            input_mask=input_mask,
            output_token=output_tokens,
            labels=labels,
            jaw_feature=jaw_features_padded,
            expression_feature=expression_features_padded,
            visual_feature_len=visual_lens,
            visual_mask=visual_features_mask
        )





    def qformer_collate_fn(self, batch: List[AVSRItem], override_drop_ratio=None):
        ####### Prepare visual features #######
        jaw_feats = [item.jaw_feat for item in batch]
        expression_feats = [item.expression_feat for item in batch]
        visual_lens = torch.tensor([jf.shape[0] for jf in jaw_feats])
        visual_feature_max_len = visual_lens.max()
        visual_features_mask = torch.zeros(len(batch), visual_feature_max_len, dtype=torch.bool)
        # Pad visual features
        jaw_features_padded = torch.zeros(len(batch), visual_feature_max_len, jaw_feats[0].shape[1])
        expression_features_padded = torch.zeros(len(batch), visual_feature_max_len, expression_feats[0].shape[1])
        for i, (jf, ef) in enumerate(zip(jaw_feats, expression_feats)):
            jaw_features_padded[i, :jf.shape[0], :] = jf
            expression_features_padded[i, :ef.shape[0], :] = ef
            visual_features_mask[i, :jf.shape[0]] = True

        ####### Prepare input and output sequneces. ############
        # Input is <bos> <start_of_visual> query_seq <start_of_audio> speech tokens
        # Output is <start_of_visual> query_seq <start_of_audio> speech tokens <eos>, loss only computed over speech tokens 

        speech_tokens = [item.speech_token for item in batch]
        speech_lens = torch.tensor([st.shape[0] for st in speech_tokens])
        query_token_lens = (visual_lens / 5).ceil().to(torch.long)
        # query token les cannot be smaller than 1 but no longer than 128 (max_query_size)
        query_token_lens = torch.max(query_token_lens, torch.ones_like(query_token_lens))
        query_token_lens = torch.min(query_token_lens, torch.ones_like(query_token_lens) * 128)

        prompt_lens = query_token_lens + speech_lens + 3 - 1# bos, start_of_visual, start_of_audio, -1 to account for shift-by-one
        prompt_lens_max = prompt_lens.max()

        output_tokens = torch.full((len(batch), prompt_lens_max), self.pad_token_id, dtype=torch.long)
        input_tokens = torch.full((len(batch), prompt_lens_max), self.pad_token_id, dtype=torch.long)
        input_mask = torch.zeros((len(batch), prompt_lens_max), dtype=torch.bool)
        query_mask = torch.zeros((len(batch), prompt_lens_max), dtype=torch.bool)
        input_tokens[:, 0] = self.bos_token_id
        input_tokens[:, 1] = self.start_of_visual_token
        
        for i, speech_token in enumerate(speech_tokens):
            cur_query_len, cur_speech_len = query_token_lens[i], speech_lens[i]
            if override_drop_ratio is not None:
                num_tokens_to_mask = int(cur_speech_len * override_drop_ratio)
            else:
                num_tokens_to_mask = int(cur_speech_len * self.drop_audio_ratio)
            if num_tokens_to_mask > 0:
                # Randomly select positions to mask
                mask_positions = torch.randperm(cur_speech_len)[:num_tokens_to_mask]
                noise_token = speech_token.clone()
                noise_token[mask_positions] = self.speech_mask_token
            else:
                noise_token = speech_token

            input_tokens[i, 2:2+cur_query_len] = self.pad_token_id # fill query positions with mask
            query_mask[i, 2:2+cur_query_len] = True
            input_tokens[i, 2+cur_query_len] = self.start_of_audio_token
            # we don't need the last token as input as we're training for NTP task
            cur_prompt_len = 2+cur_query_len+1+cur_speech_len-1
            input_tokens[i, 2+cur_query_len+1:cur_prompt_len] = noise_token[:-1]
            input_mask[i, :cur_prompt_len] = True

            # input: (bos, <sv>, query) + <sa> + speech (except last token)
            # length:      2+query_len      1     speech_len-1
            # output: (<sv> query <sa>)  + speech (with last token)
            # length:      2+query_len     speech_len
            output_tokens[i, 2+cur_query_len: 2+cur_query_len+ cur_speech_len] = speech_token
        labels = output_tokens.clone()
        labels[output_tokens == self.pad_token_id] = -100

        return AVQFormerBatch(
            keys=[item.key for item in batch],
            input_token=input_tokens,
            input_mask=input_mask,
            output_token=output_tokens,
            labels=labels,
            jaw_feature=jaw_features_padded,
            expression_feature=expression_features_padded,
            visual_feature_len=visual_lens,
            visual_mask=visual_features_mask,
            query_mask=query_mask
        )



    def qformer_avsr_collate_fn(self, batch: List[AVSRItem], override_drop_ratio=None):
        ####### Prepare visual features #######
        jaw_feats = [item.jaw_feat for item in batch]
        expression_feats = [item.expression_feat for item in batch]
        visual_lens = torch.tensor([jf.shape[0] for jf in jaw_feats])
        visual_feature_max_len = visual_lens.max()
        visual_features_mask = torch.zeros(len(batch), visual_feature_max_len, dtype=torch.bool)
        # Pad visual features
        jaw_features_padded = torch.zeros(len(batch), visual_feature_max_len, jaw_feats[0].shape[1])
        expression_features_padded = torch.zeros(len(batch), visual_feature_max_len, expression_feats[0].shape[1])
        for i, (jf, ef) in enumerate(zip(jaw_feats, expression_feats)):
            jaw_features_padded[i, :jf.shape[0], :] = jf
            expression_features_padded[i, :ef.shape[0], :] = ef
            visual_features_mask[i, :jf.shape[0]] = True

        ####### Prepare input and output sequneces. ############
        # Input is <bos> <start_of_visual> query_seq <start_of_audio> speech tokens
        # Output is <start_of_visual> query_seq <start_of_audio> speech tokens <eos>, loss only computed over speech tokens 

        speech_tokens = [item.speech_token for item in batch]
        speech_lens = torch.tensor([st.shape[0] for st in speech_tokens])
        text_tokens = [item.text_token for item in batch]
        text_lens = torch.tensor([tt.shape[0] for tt in text_tokens])

        query_token_lens = (visual_lens / 5).ceil().to(torch.long)
        # query token les cannot be smaller than 1 but no longer than 128 (max_query_size)
        query_token_lens = torch.max(query_token_lens, torch.ones_like(query_token_lens))
        query_token_lens = torch.min(query_token_lens, torch.ones_like(query_token_lens) * 128)

        query_speech_text_lens = query_token_lens + speech_lens + text_lens
        query_speech_text_lens_max = query_speech_text_lens.max()
    
        noise_speech_tokens = []
        for i, speech_token in enumerate(speech_tokens):
            cur_query_len, cur_speech_len = query_token_lens[i], speech_lens[i]
            if override_drop_ratio is not None:
                num_tokens_to_mask = int(cur_speech_len * override_drop_ratio)
            else:
                num_tokens_to_mask = int(cur_speech_len * self.drop_audio_ratio)
            if num_tokens_to_mask > 0:
                # Randomly select positions to mask
                mask_positions = torch.randperm(cur_speech_len)[:num_tokens_to_mask]
                noise_token = speech_token.clone()
                noise_token[mask_positions] = self.speech_mask_token
            else:
                noise_token = speech_token
            noise_speech_tokens.append(noise_token)

        prompt_instruction = "Transcribe the following audio visual clip:\n\n"
        prompt_token = torch.tensor(self.tokenizer.encode(prompt_instruction, add_special_tokens=False))
        prompt_len = prompt_token.shape[0]
        sample_max_len = 5 + prompt_len + query_speech_text_lens_max # 5 special tokens: bos, prompt, <start_v>, <start_a>, <start_text>
        prompts_padded = torch.full((len(batch), sample_max_len), self.pad_token_id, dtype=torch.long)
        speech_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        text_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        query_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        input_mask = torch.zeros(len(batch), sample_max_len, dtype=torch.bool)
        prompt_lens = []
        for i, (speech_token, text_token) in enumerate(zip(noise_speech_tokens, text_tokens)):
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
            input_mask[i, :curr_idx] = True
            prompt_lens.append(curr_idx + 1)


        input_tokens = prompts_padded[:, :-1]
        output_tokens = prompts_padded[:, 1:]
        target_mask = torch.zeros_like(output_tokens, dtype=torch.bool)
        for i, text_mask in enumerate(text_mask):
            target_mask[i, :] = text_mask[1:]
        labels = output_tokens.clone()
        labels[~target_mask] = -100

        return AVQFormerBatch(
            keys=[item.key for item in batch],
            input_token=input_tokens,
            input_mask=input_mask[:, :-1],
            output_token=output_tokens,
            labels=labels,
            jaw_feature=jaw_features_padded,
            expression_feature=expression_features_padded,
            visual_feature_len=visual_lens,
            visual_mask=visual_features_mask,
            query_mask=query_mask[:, :-1]
        )

