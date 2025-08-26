# different lrs_data_module/mm_data_module for AVSR finetuning, this dataloader is to prepare next-token-prediction data
# for AVLM pretraining. The target objective is the speech units

import json
import torch
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
from itertools import chain
from collections import defaultdict
from typing import List, Optional
from dataclasses import dataclass
import random
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)



@dataclass
class AVConvItem:
    question_speech_token : Optional[torch.Tensor]
    response_speech_token: Optional[torch.Tensor]
    emo_label: Optional[torch.Tensor]
    jaw_feat: torch.Tensor
    expression_feat: torch.Tensor
    key: Optional[str]


@dataclass
class AudioVisualBatch:
    keys: Optional[List[str]]
    # individual tensor and information
    jaw_feature: torch.Tensor
    expression_feature: torch.Tensor
    visual_feature_len: torch.Tensor
    visual_mask: torch.Tensor
    query_mask: torch.Tensor

    input_token: torch.Tensor
    input_mask: torch.Tensor
    output_token: torch.Tensor
    labels: torch.Tensor
    emo_label: torch.Tensor



EMO2ID = {
    "Neutral": 0,
    "Angry": 1,
    "Frustrated": 2,
    "Happy": 3,
    "Sad": 4
}


# Train Emo count: defaultdict(<class 'int'>, {'Neutral': 931, 'Angry': 647, 'Frustrated': 1048, 'Happy': 815, 'Sad': 477})
# Valid Emo count: defaultdict(<class 'int'>, {'Angry': 70, 'Frustrated': 115, 'Happy': 104, 'Neutral': 102, 'Sad': 59})
# Test Emo count: defaultdict(<class 'int'>, {'Sad': 63, 'Happy': 107, 'Neutral': 108, 'Frustrated': 131, 'Angry': 82})

class AudioVisualDataset(Dataset):
    def __init__(self, data_list, min_len=0, max_len=9999999, tokenizer=None, is_train=False, fusion_mode="x-attn"):
        assert tokenizer is not None, "SPIRITLM tokenizer must be provided"
        self.tokenizer = tokenizer
        self.fusion_mode = fusion_mode
        self.data = []
        data_len = []
        speech_len_limit = 1024
 
        num_filtered = 0
        emo_count = defaultdict(int)
        for i, item in enumerate(data_list):
            question_len = len(item['question_units'].split('['))
            response_len = len(item['response_units'].split('['))
            emo = item['emo']
            emo_count[emo] += 1
            total_len = question_len + response_len

            if is_train and total_len > speech_len_limit:
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

        question_str, response_str = item['question_units'], item['response_units']
        question_token = self.tokenizer(question_str, return_tensors="pt")['input_ids'].squeeze(0)
        response_token = self.tokenizer(response_str, return_tensors="pt")['input_ids'].squeeze(0)
        emo_label = torch.tensor(EMO2ID[item['emo']])
        smirk_feat = np.load(item['smirk_feat_path'], allow_pickle=True).item()
        # only use jaw and expression features
        jaw_feat = torch.from_numpy(smirk_feat['jaw_params']) # T x 3
        expression_feat = torch.from_numpy(smirk_feat['expression_params']) # T x 50

        return AVConvItem(
            question_speech_token=question_token,
            response_speech_token=response_token,
            emo_label=emo_label,
            jaw_feat=jaw_feat,
            expression_feat=expression_feat,
            key=key)
    



class AVLMFinetuneDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_dir, batch_size=32, num_workers=4, seed=42, fusion_mode="x-attn"):
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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_workers = 1 # for debug
        self.seed = seed
        self.fusion_mode = fusion_mode

        if self.fusion_mode == "speech_only":
            prompt_instruction = "Listen to the given audio. Determine the emotion of the speaker and continue the dialogue with the same emotion."
            
        else:
            prompt_instruction = "Perceive the given visual and audio input. Determine the emotion of the speaker and continue the dialogue with the same emotion."

        self.prompt_instruction_token = self.tokenizer.encode(prompt_instruction, add_special_tokens=False)
        self.emo_prompt = {}
        for emo in EMO2ID.keys():
            emo_prompt = f"\nEmotion: {emo}\nResponse:"
            emo_index = EMO2ID[emo]
            self.emo_prompt[emo_index] = self.tokenizer.encode(emo_prompt, add_special_tokens=False)


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
        # print data stats
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        train_collate_fn = lambda batch: self.collate_fn(batch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=BalancedEmotionSampler(self.train_dataset, self.batch_size, self.seed),
            num_workers=self.num_workers,
            collate_fn=train_collate_fn
        )

    def val_dataloader(self):
        # lengths = self.val_dataset.lengths()
        val_collate_fn = lambda batch: self.collate_fn(batch)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=val_collate_fn
        )

    def test_dataloader(self):
        test_collate_fn = lambda batch: self.collate_fn(batch)
        return DataLoader(
            self.test_dataset,
            batch_size=1, # for simplicity
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=test_collate_fn
        )
    

    def collate_fn(self, batch: List[AVConvItem]):
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

        ####### Prepare Prompt and Target Tokens #######
        question_speech_tokens = [item.question_speech_token for item in batch]
        response_speech_tokens = [item.response_speech_token for item in batch]
        emo_labels = [item.emo_label for item in batch]

        ####### Prepare Input and Output Tokens #######
        question_speech_lens = torch.tensor([st.shape[0] for st in question_speech_tokens])
        response_speech_lens = torch.tensor([st.shape[0] for st in response_speech_tokens])
        conv_lens = question_speech_lens + response_speech_lens
        prompt_instruction_len = len(self.prompt_instruction_token)
        emo_lens = []
        for i in range(len(batch)):
            cur_emotion = emo_labels[i].item()
            emo_prompt = self.emo_prompt[cur_emotion]
            emo_lens.append(len(emo_prompt))
        emo_lens = torch.tensor(emo_lens)
        conv_lens += emo_lens + prompt_instruction_len

      
        if self.fusion_mode == "speech_only":
            # Prompt: <bos> text instruction <start_of_audio> question_speech
            # Emotion: emo_label
            # Response: <start_of_audio> response_speech <eos>
            batch_prompt_len = conv_lens.max() + 4 # 4 for <bos>, 2* <start_of_audio>, <eos>
        else:
            # Prompt: <bos> text instruction <start_of_visual> visual_features <start_of_audio> question_speech
            # Emotion: emo_label
            # Response: <start_of_audio> response_speech <eos>
            query_token_lens = (visual_lens / 5).ceil().to(torch.long)
            # query token les cannot be smaller than 1 but no longer than 128 (max_query_size)
            query_token_lens = torch.max(query_token_lens, torch.ones_like(query_token_lens))
            query_token_lens = torch.min(query_token_lens, torch.ones_like(query_token_lens) * 128)
            conv_lens += query_token_lens
            batch_prompt_len = conv_lens.max() + 5 #for <bos>, <start_of_visual>, 2*<start_of_audio>, <eos>

        input_mask = torch.zeros(len(batch), batch_prompt_len, dtype=torch.bool)
        target_mask = torch.zeros_like(input_mask)
        query_mask = torch.zeros_like(input_mask)

        prompt_tokens = torch.full((len(batch), batch_prompt_len), self.pad_token_id, dtype=torch.long)
        prompt_tokens[:, 0] = self.bos_token_id
        for i in range(len(batch)):
            cur_emotion = emo_labels[i].item()
            emo_prompt = self.emo_prompt[cur_emotion]
            curr_idx = 0
            prompt_tokens[i, curr_idx] = self.bos_token_id
            curr_idx += 1
            
            prompt_tokens[i, curr_idx:curr_idx+len(self.prompt_instruction_token)] = torch.tensor(self.prompt_instruction_token, dtype=torch.long)
            curr_idx += len(self.prompt_instruction_token)
            if self.fusion_mode == "speech_only":
                prompt_tokens[i, curr_idx] = self.start_of_audio_token
                curr_idx += 1
                prompt_tokens[i, curr_idx:curr_idx+question_speech_lens[i]] = question_speech_tokens[i]
                curr_idx += question_speech_lens[i]
                prompt_tokens[i, curr_idx:curr_idx+len(emo_prompt)] = torch.tensor(emo_prompt, dtype=torch.long)
                curr_idx += len(emo_prompt)
                prompt_tokens[i, curr_idx] = self.start_of_audio_token
                curr_idx += 1
                prompt_tokens[i, curr_idx:curr_idx+response_speech_lens[i]] = response_speech_tokens[i]
                target_mask[i, curr_idx:curr_idx+response_speech_lens[i]+1] = True # +1 for <eos>
                curr_idx += response_speech_lens[i]
                prompt_tokens[i, curr_idx] = self.eos_token_id
                input_mask[i, :curr_idx+1] = True
                
            else:
                prompt_tokens[i, curr_idx] = self.start_of_visual_token
                curr_idx += 1
                prompt_tokens[i, curr_idx:curr_idx+ query_token_lens[i]] = self.pad_token_id
                query_mask[i, curr_idx:curr_idx+ query_token_lens[i]] = True

                curr_idx += query_token_lens[i]
                prompt_tokens[i, curr_idx] = self.start_of_audio_token
                curr_idx += 1
                prompt_tokens[i, curr_idx:curr_idx+question_speech_lens[i]] = question_speech_tokens[i]
                curr_idx += question_speech_lens[i]
                prompt_tokens[i, curr_idx:curr_idx+len(emo_prompt)] = torch.tensor(emo_prompt, dtype=torch.long)
                curr_idx += len(emo_prompt)
                prompt_tokens[i, curr_idx] = self.start_of_audio_token
                curr_idx += 1
                prompt_tokens[i, curr_idx:curr_idx+response_speech_lens[i]] = response_speech_tokens[i]
                target_mask[i, curr_idx:curr_idx+response_speech_lens[i]+1] = True # +1 for <eos>
                curr_idx += response_speech_lens[i]
                prompt_tokens[i, curr_idx] = self.eos_token_id
                input_mask[i, :curr_idx+1] = True
                
                
            
            # print(prompt_tokens[i])
            # decode_prompt = self.tokenizer.decode(prompt_tokens[i])
        #     print(decode_prompt)
        # exit(0)
        # shift the tokens and prepare labels and mask for training
        input_tokens = prompt_tokens[:, :-1]
        output_tokens = prompt_tokens[:, 1:]
        input_mask = input_mask[:, :-1]
        query_mask = query_mask[:, :-1]
        target_mask = target_mask[:, 1:]
        labels = output_tokens.clone()
        labels[~target_mask] = -100
        return AudioVisualBatch(
            keys=[item.key for item in batch],
            emo_label=torch.tensor(emo_labels),
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


class BalancedEmotionSampler(Sampler):
    """
    Creates batches with balanced emotion classes, ensuring each batch contains
    an approximately equal number of samples from each emotion class.
    """
    def __init__(self, dataset, batch_size, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Group indices by emotion class
        self.emotion_indices = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            emo = item['emo']
            self.emotion_indices[emo].append(idx)
            
        # Calculate class counts for reporting
        self.class_counts = {emo: len(indices) for emo, indices in self.emotion_indices.items()}
        print(f"Emotion class counts: {self.class_counts}")
        
        # Determine number of samples per class per batch
        self.num_classes = len(self.emotion_indices)
        # Ensure batch size is divisible by number of classes, rounding up if needed
        self.samples_per_class_per_batch = self.batch_size // self.num_classes

        if self.batch_size % self.num_classes != 0:
            self.samples_per_class_per_batch += 1
            self.actual_batch_size = self.samples_per_class_per_batch * self.num_classes
            print(f"Adjusted batch size to {self.actual_batch_size} to ensure class balance")
        else:
            self.actual_batch_size = self.batch_size
        
        # Determine total number of complete batches we can create
        min_class_size = min(len(indices) for indices in self.emotion_indices.values())
        max_batches_without_replacement = min_class_size // self.samples_per_class_per_batch
        
        # Number of complete batches we'll create
        self.num_batches = max(1, max_batches_without_replacement)
        self.total_samples = self.num_batches * self.actual_batch_size
        
        print(f"Creating {self.num_batches} balanced batches with {self.samples_per_class_per_batch} samples per class per batch")
        
    def __iter__(self):
        # Create random number generator with seed
        rng = random.Random(self.seed)
        
        # Shuffle indices within each emotion class
        shuffled_indices = {
            emo: rng.sample(indices, len(indices)) 
            for emo, indices in self.emotion_indices.items()
        }
        
        # Create balanced batches
        batches = []
        
        # Keep track of where we are in each emotion's list
        class_positions = {emo: 0 for emo in shuffled_indices}
        
        for _ in range(self.num_batches):
            batch = []
            
            # Add samples_per_class_per_batch from each emotion
            for emo, indices in shuffled_indices.items():
                pos = class_positions[emo]
                # If we're near the end of the list and don't have enough samples,
                # we'll cycle back to the beginning (with replacement)
                if pos + self.samples_per_class_per_batch > len(indices):
                    # Shuffle the indices again to avoid repeating the same pattern
                    shuffled_indices[emo] = rng.sample(indices, len(indices))
                    class_positions[emo] = 0
                    pos = 0
                
                # Take the next samples_per_class_per_batch samples
                batch.extend(shuffled_indices[emo][pos:pos+self.samples_per_class_per_batch])
                class_positions[emo] = pos + self.samples_per_class_per_batch
            batches.extend(batch)        

        return iter(batches)
    
    def __len__(self):
        return self.total_samples

