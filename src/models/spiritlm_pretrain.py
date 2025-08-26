import torch
import os
import jiwer
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from src.models.utils import get_attention_mask
from src.data_utils.lrs_data_module import AudioVisualBatch
from transformers import get_linear_schedule_with_warmup, LlamaForCausalLM, LlamaConfig
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from einops import repeat
EPS = 1e-12  # Epsilon to prevent division by zero


@dataclass
class BatchItem:
    input_tokens: torch.Tensor
    input_mask: torch.Tensor
    target_tokens: torch.Tensor
    target_mask: torch.Tensor
    # for av fusion
    speech_token: torch.Tensor
    speech_mask: torch.Tensor
    visual_feature: torch.Tensor
    visual_mask: torch.Tensor


class TextOnlyLogitsProcessor:
    def __call__(self, input_ids, scores):
        scores[:, 32000:] = float('-inf')  # mask non-text tokens
        return scores
            
class AVSRModel(pl.LightningModule):
    def __init__(self, hparams, tokenizer):
        """
        hparams should include:
            - audio_feature_dim: Dimensionality of the input speech features.
            - video_feature_dim: Dimensionality of the input visual features.
            - fusion_seq_len: The common sequence length to pool both modalities to.
            - d_model: The projected dimension for fusion and transformer.
            - num_emotions: Number of emotion classes.
            - learning_rate: Learning rate for the optimizer.
            - weight_decay: Weight decay for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.fusion_mode = self.hparams.fusion_mode
        self.perceiver_only = self.hparams.perceiver_only
        d_model = 4096
        spiritlm_path = os.path.join(os.environ['SPIRITLM_CHECKPOINTS_DIR'], "spiritlm_model", "spirit-lm-expressive-7b")
        if self.hparams.ckpt_path is not None: # loading from ckpt means we don't want to load pre-trained weights
            config = LlamaConfig.from_pretrained(spiritlm_path)
            spiritlm_model = LlamaForCausalLM(config=config).to(torch.bfloat16)
        else: # loading from pre-trained weights
            spiritlm_model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=spiritlm_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        # spiritlm_model.gradient_checkpointing_enable()
        self.spiritlm_model: PeftModelForCausalLM = self.inject_lora(spiritlm_model)

        self.trainable_llm_params = []
        for param in self.spiritlm_model.parameters():
            if param.requires_grad:
                self.trainable_llm_params.append(param)

        self.tokenizer = tokenizer
        self.tokenizer_vocab = tokenizer.get_vocab()
        self.start_of_av_token = self.tokenizer_vocab['[Madeuptoken32766]']
        self.start_of_audio_token = self.tokenizer_vocab['[Madeuptoken32765]']
        self.start_of_visual_token = self.tokenizer_vocab['[Madeuptoken32764]']
        self.start_of_text_token = self.tokenizer_vocab['[Madeuptoken32763]']
        self.speech_mask_token = self.tokenizer_vocab['[Madeuptoken32762]']
        # Initialize learnable dummy vector of dimension 4096
        # self.dummy_vector = nn.Parameter(torch.randn(4096, dtype=torch.bfloat16))
        # self.speech_tokenizer = spiritlm_expressive()
    
        self.trainable_fusion_params = []
        
        if not self.fusion_mode == "asr_only":
            if self.fusion_mode == "x-attn":
                self.visual_encoder = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.Linear(1024, d_model), # hard code 512 for now
                ).to(torch.bfloat16)
                self.transformer_layer = nn.TransformerDecoderLayer(
                    d_model=d_model, 
                    nhead=self.hparams.n_head, 
                    batch_first=True,
                    dtype=torch.bfloat16
                )
                
                self.perceiver = nn.TransformerDecoder(
                    self.transformer_layer, 
                    num_layers=self.hparams.n_layers, 
                )
                self.max_latents = 512
                self.latents = nn.Parameter(torch.randn(self.max_latents, d_model, dtype=torch.bfloat16))
                nn.init.normal_(self.latents, std = 0.02)
                
                for param in self.perceiver.parameters():
                    self.trainable_fusion_params.append(param)
                self.trainable_fusion_params.append(self.latents)

            elif self.fusion_mode == "concate":
                self.visual_encoder = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.Linear(1024, 512), # use its original dim
                ).to(torch.bfloat16)
                self.fusion_layer = nn.Sequential(
                    nn.Linear(4096 + 512, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, d_model)
                ).to(torch.bfloat16)
                for param in self.fusion_layer.parameters():
                    self.trainable_fusion_params.append(param)
            elif self.fusion_mode == "qformer":
                raise NotImplementedError("QFormer is not implemented yet")
            else:
                raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
            
            for param in self.visual_encoder.parameters():
                self.trainable_fusion_params.append(param)
           
        else:
            logging.info("ASR only mode, no fusion module is used")


        # Loss function.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self._val_outputs, self._test_outputs = [], []
        self.automatic_optimization = False


    def inject_lora(self, model: LlamaForCausalLM):
        # Configure LoRA for attention layers
        config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Wrap model with LoRA
        lora_model = get_peft_model(model, config)
        
        # Freeze all parameters except LoRA
        for param in lora_model.parameters():
            param.requires_grad = False  # Freeze all parameters, we use pre-trained weights
        # Unfreeze LoRA parameters
        for name, param in lora_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        return lora_model



    def compute_cross_attention(self, visual_features, item: BatchItem):
        """
        Compute cross attention between visual and speech features with proper masking
        
        Args:
            visual_features: (B, T_v, D_v) visual features
            prompt_features: (B, T_p, D_p) prompt features containing instructions, speech tokens, and text
            visual_mask: (B, T_v) boolean mask for valid visual positions
            speech_mask: (B, T_p) boolean mask for speech token positions in prompt
        """
        # prepare source info: we interleave speech and visual features
        speech_token = item.speech_token
        # this speech mask is different from item.speech_mask (which is created based on prompt_len)
        speech_mask = (speech_token != self.speech_mask_token) & (speech_token != self.tokenizer.pad_token_id)
        with torch.no_grad(): # do not update embedding of speech, only tune visual and perceiver to align with speech
            speech_repr = self.spiritlm_model.model.model.embed_tokens(speech_token)
        speech_repr[~speech_mask] = 0
        ######## Interleave Speech and Visual Features ########
        B, T_v, D_v = visual_features.shape
        interleaved_feature = visual_features.new_full((B, 2*T_v, D_v), 0) # speech and visual have the same length
        interleaved_feature[:, ::2, :] = speech_repr
        interleaved_feature[:, 1::2, :] = visual_features
        interleaved_mask = speech_mask.new_full((B, 2*T_v), False)

        interleaved_mask[:, ::2] = speech_mask
        interleaved_mask[:, 1::2] = item.visual_mask

        ####### Use Perceiver to obtain multi-modal speech tokens #######
        cur_latents = self.latents[:speech_token.shape[1]]
        cur_latents = repeat(cur_latents, 'n d -> b n d', b = B)
        T_p = cur_latents.shape[1]
        causal_mask = torch.triu(
            torch.full((T_p, T_p), float('-inf'), device=self.device),
            diagonal=1
        ).to(torch.bfloat16)
        attended_features = self.perceiver(
            tgt=cur_latents,
            tgt_is_causal=True,
            tgt_mask=causal_mask,
            memory=interleaved_feature,
            memory_key_padding_mask=~interleaved_mask,
            # memory_mask=cross_attention_mask, # no need to prepare customized cross attention mask
        ) # B, T_p, D
        return attended_features


  

    @staticmethod
    def create_index_matrix(mask, H):
        """
        Args:
        mask (torch.BoolTensor): shape (B, T) boolean mask.
        H (int): desired number of indices per row.
        
        Returns:
        idx_matrix (torch.LongTensor): shape (B, H) where each row contains the integer positions 
                                        where mask is True (in order), padded with -1 if needed.
        """
        B, T = mask.shape
        # Compute the order (i.e. sequential count) of True values along T.
        order = torch.cumsum(mask, dim=1) - 1  # shape (B, T), valid only for True positions

        # Create a tensor of indices for each position in T, broadcasted for each batch.
        indices = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)

        # Initialize the index matrix with -1 (for padding)
        idx_matrix = torch.full((B, H), -1, dtype=torch.long, device=mask.device)

        # We only want to consider true entries whose order is less than H.
        valid = mask & (order < H)

        # Get batch indices corresponding to valid True positions.
        batch_idx = torch.nonzero(valid, as_tuple=False)[:, 0]
        # The target column in idx_matrix is given by the order of the true value.
        col_idx = order[valid]
        # The values we want to store are the original indices (positions in T)
        vals = indices[valid]

        # Scatter the values into idx_matrix at the appropriate batch and column positions.
        idx_matrix[batch_idx, col_idx] = vals

        return idx_matrix

    def concate_fusion(self, speech_token, visual_features, prompt_embed, visual_mask, speech_mask):
        # B, T_v, D = visual_features.shape
        # B, T_p, D = prompt_embed.shape
        # B, T_v = visual_mask.shape
        # B, T_p = speech_mask.shape

        with torch.no_grad():
            speech_embed = self.spiritlm_model.model.model.embed_tokens(speech_token)
        av_feature = self.fusion_layer(torch.cat([visual_features, speech_embed], dim=-1)) # B, T, D
        prompt_positions = self.create_index_matrix(speech_mask, visual_mask.shape[1])
        dummy_col = torch.zeros_like(prompt_embed[:,:1], device=prompt_embed.device)
        modified_embed = torch.cat([prompt_embed, dummy_col], dim=1)
        
        # Replace embeddings for av features
        batch_indices = torch.arange(prompt_positions.shape[0], device=prompt_positions.device).unsqueeze(1)
        modified_embed[batch_indices, prompt_positions] = av_feature
        # Remove dummy column
        modified_embed = modified_embed[:,:-1]
        return modified_embed
        

    def forward(self, item: BatchItem):
        # obtain embeddings

        prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)
        if self.fusion_mode == "asr_only":
            # no need to take of visual features
            inputs_embeds = prompt_embed
        else:
            # B, T, _ = item.visual_feature.shape
            visual_features = self.visual_encoder(item.visual_feature) # B, T, D=512

            if self.fusion_mode == "x-attn":       
                B, T, D = prompt_embed.shape 
                av_feature =self.compute_cross_attention(visual_features, item) # B, T, D
                dummy_embeds = prompt_embed.new_full((B, 1, D), 0)
                clone_embeds = torch.concat([prompt_embed.clone(), dummy_embeds], dim=1)
                speech_token_max_len = av_feature.shape[1]
                speech_position = self.create_index_matrix(item.speech_mask, speech_token_max_len)
                batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
                clone_embeds[batch_indices, speech_position] = av_feature
                inputs_embeds = clone_embeds[:, :-1]

            elif self.fusion_mode == "concate":
                inputs_embeds = self.concate_fusion(item.speech_token, visual_features, prompt_embed, item.visual_mask, item.speech_mask)
            else:
                raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")

        # Get logits from spiritlm model
        labels = item.target_tokens
        logits = self.spiritlm_model(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=labels
        ).logits # B, T, Vocab_size
        return logits

    @staticmethod
    def print_model_weights(module):
        param_sum = 0.0
        for name, param in module.named_parameters():
            param_sum += param.sum().item()
        return param_sum

    def training_step(self, batch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)

        logits = self(item)

        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(-1, V), item.target_tokens.view(-1))
        preds = torch.argmax(logits, dim=-1)
        # Compute accuracy only over non-masked positions (target != -100)
        valid_mask = item.target_tokens != -100
        acc = (preds[valid_mask] == item.target_tokens[valid_mask]).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        
        # Manual optimization
        if self.fusion_mode == "asr_only" or self.perceiver_only:
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
        else:
            visual_opt, lora_opt = self.optimizers()
            visual_scheduler, lora_scheduler = self.lr_schedulers()

        self.manual_backward(loss)
        # Only update on the last accumulation step
        if (batch_idx + 1) % self.hparams.grad_accum_every == 0:
            if self.fusion_mode == "asr_only" or self.perceiver_only:
                self.clip_gradients(optimizer, 1.0, "norm") 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                self.clip_gradients(visual_opt, 1.0, "norm") 
                visual_opt.step()
                visual_scheduler.step()
                visual_opt.zero_grad()
                self.clip_gradients(lora_opt, 1.0, "norm")
                lora_opt.step()
                lora_scheduler.step()
                lora_opt.zero_grad()
        return loss


    def validation_step(self, batch: AudioVisualBatch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        logits = self(item)
        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(-1, V), item.target_tokens.view(-1))
        preds = torch.argmax(logits, dim=-1)
        valid_mask = item.target_tokens != -100
        acc = (preds[valid_mask] == item.target_tokens[valid_mask]).float().mean()
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self._val_outputs.append({
            "pred": preds,
            "target": item.target_tokens
        })
        return loss

    def on_validation_epoch_end(self):
        # compute WER btw all preds and targets
        wer_metric = []
        pred_texts = []
        target_texts = []
        
        for item in self._val_outputs:
            cur_pred = item["pred"]
            cur_target = item["target"]
            legit_mask = cur_target != -100
            cur_pred = cur_pred[legit_mask]
            cur_target = cur_target[legit_mask]
            pred_str = self.tokenizer.decode(cur_pred, skip_special_tokens=True)
            target_str = self.tokenizer.decode(cur_target, skip_special_tokens=True)
            
            pred_texts.append(pred_str)
            target_texts.append(target_str)
            
            wer = jiwer.wer(target_str, pred_str)
            wer_metric.append(wer)
            
        avg_wer = sum(wer_metric) / len(wer_metric)
        self.log("val_wer", avg_wer, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log predictions and targets to wandb if using wandb logger
        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(
                key="val_predictions",
                columns=["prediction", "target"], 
                data=[[p, t] for p, t in zip(pred_texts, target_texts)]
            )
        self._val_outputs = []

    def prepare_batch(self, batch:AudioVisualBatch):
        """
        Expects batch to have:
            keys: Optional[List[str]]
            
            # Individual tensor and information
            speech_token: Optional[torch.Tensor]
            speech_token_len: Optional[torch.Tensor] 
            speech_mask: torch.Tensor

            visual_feature: torch.Tensor
            visual_feature_len: torch.Tensor
            visual_mask: torch.Tensor

            text_token: Optional[torch.Tensor]
            text_token_len: Optional[torch.Tensor]
            text_mask: torch.Tensor

            # Prompt and prompt length
            prompt_token: torch.Tensor
            prompt_token_len: torch.Tensor
        """

        # Get input and target sequences using shift-by-1 strategy
        input_tokens = batch.prompt_token[:, :-1]  # Remove last token for input
        target_tokens = batch.prompt_token[:, 1:]  # Remove first token for target
        
        # Create attention mask for input sequence
        # Mask should be 1 for valid tokens, 0 for padding
        input_mask = torch.zeros_like(input_tokens, dtype=torch.bool, device=self.device)
        for i, length in enumerate(batch.prompt_token_len - 1):  # -1 since we removed last token
            input_mask[i, :length] = 1
            
        # Create loss mask that matches target sequence length
        target_mask = torch.zeros_like(target_tokens, dtype=torch.bool, device=self.device) 
        # we only use the text token part for target loss computation
        # prompt:    [a] [b] [c] [start_text] [t1] [t2] [t3] [eos] [pad]
        # text_mask: [0] [0] [0]     [0]       [1]  [1]  [1] [0]   [0]
        # input:     [a] [b] [c] [start_text] [t1] [t2] [t3] [eos]
        # input_mask: [1] [1] [1]     [1]     [1]  [1]  [1]  [1]
        # target:    [b] [c] [start_text] [t1] [t2] [t3] [eos] [pad]
        # target_mask:[0] [0]   [0]       [1]  [1]  [1] [0/1] [0]
        for i, text_mask in enumerate(batch.text_mask):
            target_mask[i, :] = text_mask[1:]
        labels = target_tokens.clone()
        labels[~target_mask] = -100


        speech_dropout_mask = input_tokens == self.speech_mask_token
        return BatchItem(
            input_tokens=input_tokens,
            input_mask=input_mask,
            target_tokens=labels,
            target_mask=target_mask,
            speech_token=batch.speech_token,
            speech_mask=batch.speech_mask[:, :-1],
            visual_feature=batch.visual_feature.to(torch.bfloat16),
            visual_mask=batch.visual_mask
        )
    

    def test_attn(self, item: BatchItem):
        batch_size = item.input_tokens.shape[0]
        # Find both start_of_av and start_of_text positions
        text_start_positions = []
        av_start_positions = []
        max_end_pos = 0
        
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (item.input_tokens[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 1
            
            # Find start_of_av position
            av_start_pos = (item.input_tokens[i] == self.start_of_av_token).nonzero(as_tuple=True)[0]
            av_pos = av_start_pos[-1].item() + 1
            
            text_start_positions.append(text_end_pos)
            av_start_positions.append(av_pos)
            max_end_pos = max(max_end_pos, text_end_pos)
        
        # Prepare padded input batch and attention mask as before
        input_ids = torch.full((batch_size, max_end_pos), 
                              self.tokenizer.pad_token_id, 
                              dtype=torch.long, 
                              device=item.input_tokens.device)
        
        attention_mask = torch.zeros((batch_size, max_end_pos),
                                   dtype=torch.long,
                                   device=item.input_tokens.device)
        
        # Create speech mask (True for speech token positions)
        speech_mask = torch.zeros((batch_size, max_end_pos),
                                dtype=torch.bool,
                                device=item.input_tokens.device)
        
        
        # Fill in sequences and masks
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            av_pos = av_start_positions[i]
            
            # Fill input_ids
            input_ids[i, -end_pos:] = item.input_tokens[i, :end_pos]
            
            # Fill attention mask for valid positions
            attention_mask[i, -end_pos:] = 1
            
            # Fill speech mask for positions between start_of_av and start_of_text
            speech_start = max_end_pos - end_pos + av_pos  # adjust for left padding
            speech_end = max_end_pos - end_pos + text_start_positions[i] - 1  # -1 to exclude start_of_text token
            speech_mask[i, speech_start:speech_end] = True





        # cross attention with visual features
        B, T, _ = item.visual_feature.shape
        visual_features = item.visual_feature.view(B*T, 18, 32, 32)
        visual_features = self.visual_encoder(visual_features).view(B, T, -1) # [B, T, D]
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(input_ids)
        av_feature =self.compute_cross_attention(visual_features, prompt_embed, item.visual_mask, speech_mask) # B, T, D
        input_embeds = prompt_embed.clone()
        input_embeds[speech_mask] = av_feature[speech_mask]
        # Generate with beam search
        outputs = self.spiritlm_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,  # Add attention mask
            max_length=max_end_pos + 256,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[TextOnlyLogitsProcessor()],
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        
        # Process outputs for each sequence in batch
        predictions = []
        targets = []
        
        for i in range(batch_size):
            pred_tokens = outputs[i]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            
            # Get targets
            legit_target_token = item.target_tokens[i][item.target_mask[i]]
            target_text = self.tokenizer.decode(legit_target_token, skip_special_tokens=True)

            targets.append(target_text)
  
        self._test_outputs.append({
            "predictions": predictions,
            "targets": targets
        })


    def test_concate(self, item: BatchItem):
        batch_size = item.input_tokens.shape[0]
        # first create the speech indices for fusion
        visual_features = self.visual_encoder(item.visual_feature)
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)
        fused_embeds = self.concate_fusion(item.speech_token, visual_features, prompt_embed, item.visual_mask, item.speech_mask) # same shape as input tokens
        # now we start to left pad the input for generation
        text_start_positions = []
        max_end_pos = 0
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (item.input_tokens[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 1    
            text_start_positions.append(text_end_pos)
            max_end_pos = max(max_end_pos, text_end_pos)

        # Prepare padded input batch and attention mask as before
        inputs_embeds = fused_embeds[:, :max_end_pos, :]
        left_pad_embeds = self.spiritlm_model.model.model.embed_tokens(torch.full((batch_size, max_end_pos), self.tokenizer.pad_token_id, dtype=torch.long, device=item.input_tokens.device))
        # Left pad each sequence in inputs_embeds
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            left_pad_embeds[i, max_end_pos-end_pos:max_end_pos] = inputs_embeds[i, :end_pos]
        
        attention_mask = torch.zeros((batch_size, max_end_pos), device=item.input_tokens.device)
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            attention_mask[i, max_end_pos-end_pos:max_end_pos] = 1
        outputs = self.spiritlm_model.generate(
            inputs_embeds=left_pad_embeds,
            attention_mask=attention_mask,  # Add attention mask
            max_length=max_end_pos + 256,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[TextOnlyLogitsProcessor()],
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        
        # Process outputs for each sequence in batch
        predictions = []
        targets = []
        
        for i in range(batch_size):
            pred_tokens = outputs[i]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            
            # Get targets
            legit_target_token = item.target_tokens[i][item.target_mask[i]]
            target_text = self.tokenizer.decode(legit_target_token, skip_special_tokens=True)
            targets.append(target_text)
  
        self._test_outputs.append({
            "predictions": predictions,
            "targets": targets
        })



    def test_asr_only(self, item: BatchItem):
        batch_size = item.input_tokens.shape[0]
        # first create the speech indices for fusion
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)
        text_start_positions = []
        max_end_pos = 0
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (item.input_tokens[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 2
            text_start_positions.append(text_end_pos)
            max_end_pos = max(max_end_pos, text_end_pos)

        left_pad_embeds = self.spiritlm_model.model.model.embed_tokens(torch.full((batch_size, max_end_pos), self.tokenizer.pad_token_id, dtype=torch.long, device=item.input_tokens.device))
        # Left pad each sequence in inputs_embeds
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            left_pad_embeds[i, max_end_pos-end_pos:max_end_pos] = prompt_embed[i, :end_pos]
        

        attention_mask = torch.zeros((batch_size, max_end_pos), device=item.input_tokens.device)
        drop_position = item.input_tokens == self.speech_mask_token
        num_drop = drop_position.sum()

        for i in range(batch_size):
            end_pos = text_start_positions[i]
            if num_drop > 0:
                attention_mask[i, max_end_pos-end_pos:max_end_pos] = ~drop_position[i, :end_pos]
            else:
                attention_mask[i, max_end_pos-end_pos:max_end_pos] = 1

        outputs = self.spiritlm_model.generate(
            inputs_embeds=left_pad_embeds,
            attention_mask=attention_mask,  # Add attention mask
            max_length=max_end_pos + 128,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[TextOnlyLogitsProcessor()],
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        
        # Process outputs for each sequence in batch
        predictions = []
        targets = []
        
        for i in range(batch_size):
            # Get targets
            legit_target_token = item.target_tokens[i][item.target_mask[i]]
            target_text = self.tokenizer.decode(legit_target_token, skip_special_tokens=True)
            targets.append(target_text)

            num_target = legit_target_token.shape[0]
            pred_tokens = outputs[i]
            pre_token = item.input_tokens[i][text_start_positions[i]-1]
            pred_tokens = torch.cat([pre_token.unsqueeze(0), pred_tokens])
            pred_tokens = pred_tokens[:num_target]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
        
        self._test_outputs.append({
            "predictions": predictions,
            "targets": targets
        })




    def test_step(self, batch: AudioVisualBatch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        if self.fusion_mode == "x-attn":
            self.test_attn(item)
        elif self.fusion_mode == "concate":
            self.test_concate(item)
        elif self.fusion_mode == "asr_only":
            self.test_asr_only(item)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
       
        
       

    def on_test_epoch_end(self):
        # Aggregate all predictions and targets
        all_preds = []
        all_targets = []
        for output in self._test_outputs:
            all_preds.extend(output["predictions"])
            all_targets.extend(output["targets"])
        
        # Compute metrics
        wer = jiwer.wer(all_targets, all_preds)
        self.log("test_wer", wer, sync_dist=True)
        


    def configure_optimizers(self):
        # Collect parameters for visual encoder and transformer decoder
        visual_params = self.trainable_fusion_params
        llm_params = self.trainable_llm_params
        if self.fusion_mode == "asr_only":
            # only have one optimizer for llm parameters
            optimizer = torch.optim.Adam(
                llm_params,
                lr=3e-5,  # e.g. 1e-4
                weight_decay=self.hparams.weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.max_steps
            )
            return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}]
    
        if self.perceiver_only:
            # only have one optimizer for perceiver parameters
            optimizer = torch.optim.Adam(
                visual_params,
                lr=1e-4,  # e.g. 1e-4
                weight_decay=self.hparams.weight_decay
            )
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.max_steps
            )
            return [{"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}]
        # Create optimizer for visual parameters
        visual_optimizer = torch.optim.Adam(
            visual_params,
            lr=self.hparams.learning_rate,  # e.g. 3e-4
            weight_decay=self.hparams.weight_decay
        )

        # Create optimizer for LoRA parameters
        lora_optimizer = torch.optim.Adam(
            llm_params,
            lr=self.hparams.lora_lr,
            weight_decay=self.hparams.weight_decay
        )

        # Create warmup scheduler for visual optimizer
        visual_scheduler = get_linear_schedule_with_warmup(
            visual_optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.max_steps
        )

        # Create warmup scheduler for LoRA optimizer
        lora_scheduler = get_linear_schedule_with_warmup(
            lora_optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.max_steps
        )

        return [
            {
                'optimizer': visual_optimizer,
                'lr_scheduler': {
                    'scheduler': visual_scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            },
            {
                'optimizer': lora_optimizer,
                'lr_scheduler': {
                    'scheduler': lora_scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]


class ASRModel(pl.LightningModule):
    def __init__(self, hparams, tokenizer):
        """
        same as AVSR Model, but without visual encoder and transformer decoder
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        d_model = self.hparams.d_model
        spiritlm_path = os.path.join(os.environ['SPIRITLM_CHECKPOINTS_DIR'], "spiritlm_model", "spirit-lm-expressive-7b")
        spiritlm_model = LlamaForCausalLM.from_pretrained(
            spiritlm_path,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        # spiritlm_model.gradient_checkpointing_enable()
        self.spiritlm_model: PeftModelForCausalLM = self.inject_lora(spiritlm_model)

        self.tokenizer = tokenizer
        # self.speech_tokenizer = spiritlm_expressive()

        # Loss function.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self._val_outputs, self._test_outputs = [], []
        self.automatic_optimization = False

        self.tokenizer_vocab = tokenizer.get_vocab()
        self.start_of_av_token = self.tokenizer_vocab['[Madeuptoken32766]']
        # for A->T, we have another special token to indicate that visual modality is not fused into it
        self.start_of_audio_token = self.tokenizer_vocab['[Madeuptoken32765]']
        # for V->, we have another special token to indicate that audio modality is not fused into it
        self.start_of_visual_token = self.tokenizer_vocab['[Madeuptoken32764]']
        self.start_of_text_token = self.tokenizer_vocab['[Madeuptoken32763]']
        self.speech_mask_token = self.tokenizer_vocab['[Madeuptoken32762]']

    def inject_lora(self, model: LlamaForCausalLM):
        # Configure LoRA for attention layers
        config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "embed_tokens"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Wrap model with LoRA
        lora_model = get_peft_model(model, config)
        
        # Freeze all parameters except LoRA
        for param in lora_model.parameters():
            param.requires_grad = False  # Freeze all parameters
            
        # Unfreeze LoRA parameters
        for name, param in lora_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            if "lm_head" in name:
                param.requires_grad = True
        return lora_model


    def forward(self, item: BatchItem):
        # obtain embeddings
        if isinstance(self.spiritlm_model, LlamaForCausalLM):
            prompt_embed = self.spiritlm_model.model.embed_tokens(item.input_tokens)
        else:
            prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)
        # Get logits from spiritlm model
        labels = item.target_tokens
        logits = self.spiritlm_model(
            inputs_embeds=prompt_embed,
            return_dict=True,
            labels=labels
        ).logits # B, T, Vocab_size
        return logits


    def training_step(self, batch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        logits = self(item)
        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(-1, V), item.target_tokens.view(-1))
        preds = torch.argmax(logits, dim=-1)
        legit_mask = item.target_tokens != -100
        acc = (preds[legit_mask] == item.target_tokens[legit_mask]).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        
        # Manual optimization
        lora_opt = self.optimizers()
        lora_scheduler = self.lr_schedulers()
        self.manual_backward(loss)
        # Only update on the last accumulation step
        if (batch_idx + 1) % self.hparams.grad_accum_every == 0:
            self.clip_gradients(lora_opt, 1.0, "norm")
            lora_opt.step()
            lora_scheduler.step()
            lora_opt.zero_grad()
        return loss


    def validation_step(self, batch: AudioVisualBatch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        logits = self(item)
        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(-1, V), item.target_tokens.view(-1))
        preds = torch.argmax(logits, dim=-1)
        legit_mask = item.target_tokens != -100
        acc = (preds[legit_mask] == item.target_tokens[legit_mask]).float().mean()
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self._val_outputs.append({
            "pred": preds,
            "target": item.target_tokens
        })
        return loss

    def on_validation_epoch_end(self):
        # compute WER btw all preds and targets
        wer_metric = []
        pred_texts = []
        target_texts = []
        
        for item in self._val_outputs:
            cur_pred = item["pred"]
            cur_target = item["target"]
            legit_mask = cur_target != -100
            cur_pred = cur_pred[legit_mask]
            cur_target = cur_target[legit_mask]
            pred_str = self.tokenizer.decode(cur_pred, skip_special_tokens=True)
            target_str = self.tokenizer.decode(cur_target, skip_special_tokens=True)
            
            pred_texts.append(pred_str)
            target_texts.append(target_str)
            
            wer = jiwer.wer(target_str, pred_str)
            wer_metric.append(wer)
            
        avg_wer = sum(wer_metric) / len(wer_metric)
        self.log("val_wer", avg_wer, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Log predictions and targets to wandb if using wandb logger
        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(
                key="val_predictions",
                columns=["prediction", "target"], 
                data=[[p, t] for p, t in zip(pred_texts, target_texts)]
            )
        self._val_outputs = []


    def test_step(self, batch: AudioVisualBatch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        batch_size = item.input_tokens.shape[0]
        
        # Find both start_of_av and start_of_text positions
        text_start_positions = []
        av_start_positions = []
        max_end_pos = 0
        
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (item.input_tokens[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 1
            
            # Find start_of_av position
            av_start_pos = (item.input_tokens[i] == self.start_of_av_token).nonzero(as_tuple=True)[0]
            av_pos = av_start_pos[-1].item() + 1
            
            text_start_positions.append(text_end_pos)
            av_start_positions.append(av_pos)
            max_end_pos = max(max_end_pos, text_end_pos)
        
        # Prepare padded input batch and attention mask as before
        input_ids = torch.full((batch_size, max_end_pos), 
                              self.tokenizer.pad_token_id, 
                              dtype=torch.long, 
                              device=item.input_tokens.device)
        
        attention_mask = torch.zeros((batch_size, max_end_pos),
                                   dtype=torch.long,
                                   device=item.input_tokens.device)
        
        # Create speech mask (True for speech token positions)
        speech_mask = torch.zeros((batch_size, max_end_pos),
                                dtype=torch.bool,
                                device=item.input_tokens.device)
        
        # Fill in sequences and masks
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            av_pos = av_start_positions[i]
            
            # Fill input_ids
            input_ids[i, -end_pos:] = item.input_tokens[i, :end_pos]
            
            # Fill attention mask for valid positions
            attention_mask[i, -end_pos:] = 1
            
            # Fill speech mask for positions between start_of_av and start_of_text
            speech_start = max_end_pos - end_pos + av_pos  # adjust for left padding
            speech_end = max_end_pos - end_pos + text_start_positions[i] - 1  # -1 to exclude start_of_text token
            speech_mask[i, speech_start:speech_end] = True
        
        class TextOnlyLogitsProcessor:
            def __call__(self, input_ids, scores):
                scores[:, 32000:] = float('-inf')  # mask non-text tokens
                return scores
        

        # Generate with beam search
        outputs = self.spiritlm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,  # Add attention mask
            max_length=max_end_pos + 256,
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[TextOnlyLogitsProcessor()],
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        
        # Process outputs for each sequence in batch
        predictions = []
        targets = []
        
        for i in range(batch_size):
            # Get predictions - adjust for left padding
            pred_tokens = outputs[i][max_end_pos:]  # Extract from correct position
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predictions.append(pred_text)
            
            # Get targets
            legit_target_token = item.target_tokens[i][item.target_mask[i]]
            target_text = self.tokenizer.decode(legit_target_token, skip_special_tokens=True)
            targets.append(target_text)
        
        self._test_outputs.append({
            "predictions": predictions,
            "targets": targets
        })

    def on_test_epoch_end(self):
        # Aggregate all predictions and targets
        all_preds = []
        all_targets = []
        for output in self._test_outputs:
            all_preds.extend(output["predictions"])
            all_targets.extend(output["targets"])
        
        # Compute metrics
        wer = jiwer.wer(all_targets, all_preds)
        self.log("test_wer", wer, sync_dist=True)
        


    def prepare_batch(self, batch:AudioVisualBatch):
        """
        Expects batch to have:
            keys: Optional[List[str]]
            
            # Individual tensor and information
            speech_token: Optional[torch.Tensor]
            speech_token_len: Optional[torch.Tensor] 
            speech_mask: torch.Tensor

            visual_feature: torch.Tensor
            visual_feature_len: torch.Tensor
            visual_mask: torch.Tensor

            text_token: Optional[torch.Tensor]
            text_token_len: Optional[torch.Tensor]
            text_mask: torch.Tensor

            # Prompt and prompt length
            prompt_token: torch.Tensor
            prompt_token_len: torch.Tensor
        """

        # Get input and target sequences using shift-by-1 strategy
        input_tokens = batch.prompt_token[:, :-1]  # Remove last token for input
        target_tokens = batch.prompt_token[:, 1:]  # Remove first token for target
        
        # Create attention mask for input sequence
        # Mask should be 1 for valid tokens, 0 for padding
        input_mask = torch.zeros_like(input_tokens, dtype=torch.bool, device=self.device)
        for i, length in enumerate(batch.prompt_token_len - 1):  # -1 since we removed last token
            input_mask[i, :length] = 1
            
        # Create loss mask that matches target sequence length
        target_mask = torch.zeros_like(target_tokens, dtype=torch.bool, device=self.device) 
        # we only use the text token part for target loss computation
        # prompt:    [a] [b] [c] [start_text] [t1] [t2] [t3] [eos] [pad]
        # text_mask: [0] [0] [0]     [0]       [1]  [1]  [1] [0]   [0]
        # input:     [a] [b] [c] [start_text] [t1] [t2] [t3] [eos]
        # input_mask: [1] [1] [1]     [1]     [1]  [1]  [1]  [1]
        # target:    [b] [c] [start_text] [t1] [t2] [t3] [eos] [pad]
        # target_mask:[0] [0]   [0]       [1]  [1]  [1] [0/1] [0]
        for i, text_mask in enumerate(batch.text_mask):
            target_mask[i, :] = text_mask[1:]
        labels = target_tokens.clone()
        labels[~target_mask] = -100
        return BatchItem(
            input_tokens=input_tokens,
            input_mask=input_mask,
            target_tokens=labels,
            target_mask=target_mask,
            speech_mask=batch.speech_mask[:, :-1],
            visual_feature=batch.visual_feature.to(torch.bfloat16),
            visual_mask=batch.visual_mask
        )
    

    def configure_optimizers(self):
        # Collect parameters for visual encoder and transformer decoder
        lora_params = []
        for name, param in self.spiritlm_model.named_parameters():
            if "lora" in name or "lm_head" in name:

                lora_params.append(param)

        lora_optimizer = torch.optim.Adam(
            lora_params,
            lr=self.hparams.learning_rate,  # Lower learning rate for LoRA, e.g. 1e-5
            weight_decay=self.hparams.weight_decay
        )
        
        lora_scheduler = get_linear_schedule_with_warmup(
            lora_optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.max_steps
        )
        return [
            {
                'optimizer': lora_optimizer,
                'lr_scheduler': {
                    'scheduler': lora_scheduler,
                    'interval': 'step', 
                    'frequency': 1
                }
            }
        ]
    