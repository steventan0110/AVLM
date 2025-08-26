import torch
import os
import jiwer
import json
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from src.models.utils import VisualEncoder2D, get_attention_mask
from src.data_utils.pretrain_data_util import AudioVisualBatch, AVQFormerBatch

from transformers import get_linear_schedule_with_warmup, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from src.models.qformer_utils import PerceiverResampler
from src.models.mm_avlm import SMIRKFeatureEncoder
from src.exp.spiritlm.spiritlm.model.spiritlm_model import Spiritlm
from src.exp.spiritlm.spiritlm.speech_tokenizer import spiritlm_expressive
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM

eps=1e-12	# to prevent dividing by zero


@dataclass
class BatchItem:
    input_tokens: torch.Tensor
    input_mask: torch.Tensor
    target_tokens: torch.Tensor
    target_mask: torch.Tensor
    # for av fusion
    speech_mask: torch.Tensor
    visual_feature: torch.Tensor
    visual_mask: torch.Tensor


class TextOnlyLogitsProcessor:
    def __call__(self, input_ids, scores):
        scores[:, 32000:] = float('-inf')  # mask non-text tokens
        return scores
    
class AVLMModel(pl.LightningModule):
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
        d_model = self.hparams.d_model
        spiritlm_path = os.path.join(os.environ['SPIRITLM_CHECKPOINTS_DIR'], "spiritlm_model", "spirit-lm-expressive-7b")
        self.visual_only_training = False
        if self.hparams.ckpt_path is not None: # loading from ckpt means we don't want to load pre-trained weights
            config = LlamaConfig.from_pretrained(spiritlm_path)
            spiritlm_model = LlamaForCausalLM(config=config).to(torch.bfloat16)
            # self.visual_only_training = True # add true here if we want to disable LLM tuning
        else:
            spiritlm_model = LlamaForCausalLM.from_pretrained(
                spiritlm_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        print(f"Visual only training: {self.visual_only_training}")
        # spiritlm_model.gradient_checkpointing_enable()
        self.spiritlm_model: PeftModelForCausalLM = self.inject_lora(spiritlm_model, is_qformer=self.hparams.fusion_mode == "qformer")
        self.llm_params = []
        for params in self.spiritlm_model.parameters():
            if params.requires_grad:
                self.llm_params.append(params)


        self.tokenizer = tokenizer
        self.tokenizer_vocab = tokenizer.get_vocab()
        self.start_of_av_token = self.tokenizer_vocab['[Madeuptoken32766]']
        self.start_of_audio_token = self.tokenizer_vocab['[Madeuptoken32765]']
        self.start_of_visual_token = self.tokenizer_vocab['[Madeuptoken32764]']
        self.start_of_text_token = self.tokenizer_vocab['[Madeuptoken32763]']
        self.speech_mask_token = self.tokenizer_vocab['[Madeuptoken32762]']

        self.visual_params = []
        if self.hparams.fusion_mode == "asr_only":
            print("No visual encoder as ASR only")
        elif self.hparams.fusion_mode == "qformer" or self.hparams.fusion_mode == "qformer_avsr":
            print("QFormer Mode Used")
            self.max_latents = 128
            self.visual_encoder = SMIRKFeatureEncoder()
            self.perceiver = PerceiverResampler(
                max_latents=self.max_latents,
                dim=d_model,
                depth=self.hparams.n_layers,
                dim_context=256,
                heads=32,
                dim_head=128).to(torch.bfloat16)
            for param in self.perceiver.parameters():
                self.visual_params.append(param)
            for param in self.visual_encoder.parameters():
                self.visual_params.append(param)
        elif self.hparams.fusion_mode == "qformer_infill":
            print("QFormer Infill Mode Used")
            self.max_latents = 512
            self.visual_encoder = SMIRKFeatureEncoder()
            self.perceiver = PerceiverResampler(
                max_latents=self.max_latents,
                dim=d_model,
                depth=self.hparams.n_layers,
                dim_context=256,
                heads=32,
                dim_head=128).to(torch.bfloat16)
            for param in self.perceiver.parameters():
                self.visual_params.append(param)
            for param in self.visual_encoder.parameters():
                self.visual_params.append(param)
        elif self.hparams.fusion_mode == "concate":
            print("Concate Mode Used")
            self.visual_encoder = SMIRKFeatureEncoder()
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model + 256, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model)
            ).to(torch.bfloat16)
            for param in self.fusion_layer.parameters():
                self.visual_params.append(param)
            for param in self.visual_encoder.parameters():
                self.visual_params.append(param)
        else:
            raise ValueError(f"Invalid fusion mode: {self.hparams.fusion_mode}")

        
        # Loss function.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self._val_outputs, self._test_outputs = [], []
        self.automatic_optimization = False


    def inject_lora(self, model: LlamaForCausalLM, is_qformer: bool):
        # Configure LoRA for attention layers
        config = LoraConfig(
            r=64,
            lora_alpha=128,
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



    def forward(self, input_tokens, labels, item: AudioVisualBatch):
        attention_mask, drop_mask = None, None

        drop_positions = input_tokens == self.speech_mask_token            
        if drop_positions.sum() > 0:
            # print(f"Dropping {drop_positions.sum()} positions")
            drop_mask = (~drop_positions) & (item.labels != -100)

        prompt_embed = self.spiritlm_model.model.model.embed_tokens(input_tokens)
        B, T, D = prompt_embed.shape 

        if self.hparams.fusion_mode == "qformer" or self.hparams.fusion_mode == "qformer_avsr":
            visual_feature = self.visual_encoder(item.jaw_feature, item.expression_feature).to(torch.bfloat16)
            num_query_tokens = item.query_mask.sum(dim=-1)
            max_query = num_query_tokens.max()
            query_positions = self.create_index_matrix(item.query_mask, max_query)
            qformer_output = self.perceiver(visual_feature, max_query, mask=item.visual_mask) # B, Max_Query, D

            dummy_embed = prompt_embed.new_full((B, 1, D), 0)
            clone_embeds = torch.cat([prompt_embed, dummy_embed], dim=1)
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
            clone_embeds[batch_indices, query_positions] = qformer_output
            inputs_embeds = clone_embeds[:, :-1]

            # we disable the attention over the dropped positions since visual cue is already in prefix
            attention_mask = torch.ones_like(input_tokens)
            attention_mask[drop_positions] = 0

        elif self.hparams.fusion_mode == "qformer_infill":
            visual_feature = self.visual_encoder(item.jaw_feature, item.expression_feature).to(torch.bfloat16)
            drop_indices = self.create_index_matrix(drop_positions, T)
            # we use qformer's all latents to perform cross-attn retrieval but then take out drop indices for infilling
            qformer_output = self.perceiver(visual_feature, T, mask=item.visual_mask) # B, Max_Query, D
            
            dummy_embed = prompt_embed.new_full((B, 1, D), 0)
            clone_embeds = torch.cat([prompt_embed, dummy_embed], dim=1)
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1)

            clone_embeds[batch_indices, drop_indices] = qformer_output
            inputs_embeds = clone_embeds[:, :-1]

        elif self.hparams.fusion_mode == "concate":
            visual_feature = self.visual_encoder(item.jaw_feature, item.expression_feature).to(torch.bfloat16)
            # input token is <bos> speech token (exclude the last one), so we also remove last visual frame
            speech_feature = prompt_embed[:, 1:]
            fusion_feature = torch.cat([speech_feature, visual_feature[:, :-1]], dim=-1)
            fusion_feature = self.fusion_layer(fusion_feature)

            speech_mask = item.input_mask
            speech_mask[:, 0] = False
            speech_indices = self.create_index_matrix(speech_mask, T-1)
            dummy_embed = prompt_embed.new_full((B, 1, D), 0)
            clone_embeds = torch.cat([prompt_embed, dummy_embed], dim=1)
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
            clone_embeds[batch_indices, speech_indices] = fusion_feature
            inputs_embeds = clone_embeds[:, :-1]
            assert drop_mask is None, "Drop mask should be None for concate mode"
        else:
            # speech-only mode
            inputs_embeds = prompt_embed

        logits = self.spiritlm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels
        ).logits # B, T, Vocab_size
        return logits, drop_mask
    

    @staticmethod
    def print_model_weights(module):
        param_sum = 0.0
        for _, param in module.named_parameters():
            param_sum += param.sum().item()
        return param_sum


    def training_step(self, batch: AudioVisualBatch, batch_idx):
        logits, drop_mask = self.forward(batch.input_token, batch.labels, batch)

        # llm_param_sum = self.print_model_weights(self.spiritlm_model.model.model)
        # perceiver_param_sum = self.print_model_weights(self.visual_encoder)
        # print(f"LLM params: {llm_param_sum}, Perceiver params: {perceiver_param_sum}")

        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        if drop_mask is not None and drop_mask.any():
            target_mask = drop_mask
        else:
            target_mask = batch.labels != -100

        valid_logits = logits[target_mask]
        valid_labels = batch.labels[target_mask]

        loss = self.loss_fn(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=-1)
        acc = (preds == valid_labels).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        
        # Manual optimization
        if self.hparams.fusion_mode == "asr_only" or self.visual_only_training:
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
        else:
            visual_opt, lora_opt = self.optimizers()
            visual_scheduler, lora_scheduler = self.lr_schedulers()
        self.manual_backward(loss)
        # Only update on the last accumulation step
      
        if (batch_idx + 1) % self.hparams.grad_accum_every == 0:
            if self.hparams.fusion_mode == "asr_only" or self.visual_only_training:
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
        labels = batch.labels
        logits, drop_mask = self(batch.input_token, labels, batch)

        B, T, V = logits.shape
        if drop_mask is not None and drop_mask.any():
            target_mask = drop_mask
        else:
            target_mask = labels != -100

        valid_logits = logits[target_mask]
        valid_labels = labels[target_mask]

        loss = self.loss_fn(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=-1)
        acc = (preds == valid_labels).float().mean()
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        # compute perplexity
        perplexity = torch.exp(loss)
        self.log("val_ppl", perplexity, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        return loss


    def test_qformer_avsr(self, batch: AVQFormerBatch):
        batch_size = batch.input_token.shape[0]
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(batch.input_token)
        B, T, D = prompt_embed.shape 
        visual_feature = self.visual_encoder(batch.jaw_feature, batch.expression_feature).to(torch.bfloat16)
        num_query_tokens = batch.query_mask.sum(dim=-1)
        max_query = num_query_tokens.max()
        query_positions = self.create_index_matrix(batch.query_mask, max_query)
        qformer_output = self.perceiver(visual_feature, max_query, mask=batch.visual_mask) # B, Max_Query, D

        dummy_embed = prompt_embed.new_full((B, 1, D), 0)
        clone_embeds = torch.cat([prompt_embed, dummy_embed], dim=1)
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
        clone_embeds[batch_indices, query_positions] = qformer_output
        inputs_embeds = clone_embeds[:, :-1]


        text_start_positions = []
        max_end_pos = 0
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (batch.input_token[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 2 # should be +1, +2 we expose it to the first text token
            text_start_positions.append(text_end_pos)
            max_end_pos = max(max_end_pos, text_end_pos)


        # Prepare padded input batch and attention mask as before
        left_pad_embeds = inputs_embeds.new_full((batch_size, max_end_pos, D), 0)
        # Left pad each sequence in inputs_embeds
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            left_pad_embeds[i, max_end_pos-end_pos:max_end_pos] = inputs_embeds[i, :end_pos]

        attention_mask = torch.zeros((batch_size, max_end_pos), device=batch.input_token.device)
        drop_position = batch.input_token == self.speech_mask_token
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

        predictions = []
        targets = []
        for i in range(batch_size):
            target = batch.labels[i]
            target_mask = target != -100
            legit_target_token = target[target_mask]
            target_text = self.tokenizer.decode(legit_target_token, skip_special_tokens=True)

            num_target = legit_target_token.shape[0]
            pred_tokens = outputs[i]
            pre_token = batch.input_token[i][text_start_positions[i]-1]
            
            pred_tokens = torch.cat([pre_token.unsqueeze(0), pred_tokens])
            pred_tokens = pred_tokens[:num_target]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)

            cur_wer = jiwer.wer(target_text, pred_text)
            if cur_wer > 0.5:
                print(f"pred: {pred_text}, target: {target_text}")

                # skip the sample that is purely hallucination

            predictions.append(pred_text)
            targets.append(target_text)

  
        self._test_outputs.append({
            "predictions": predictions,
            "targets": targets
        })


    def test_step(self, batch: AudioVisualBatch, batch_idx):
        if self.hparams.fusion_mode == "qformer_avsr":
            self.test_qformer_avsr(batch)
            return

        logits, drop_mask = self.forward(batch.input_token, batch.labels, batch)
        B, T, V = logits.shape

        # Use drop_mask instead of target-based mask
        if drop_mask is not None and drop_mask.any():
            # Flatten the drop_mask to match logits_flat and labels_flat
            mask = drop_mask
        else:
            mask = batch.labels != -100
        
        # Select only the logits and labels at positions where drop_mask is True
        valid_dropped_logits = logits[mask]
        valid_dropped_labels = batch.labels[mask]
        loss = self.loss_fn(valid_dropped_logits, valid_dropped_labels)
        perplexity = torch.exp(loss)
        
        self._test_outputs.append({
            "loss": loss,
            "perplexity": perplexity,

        })


    def on_test_epoch_end(self):
        if self.hparams.fusion_mode == "qformer_avsr":
            all_preds = []
            all_targets = []
            for output in self._test_outputs:
                all_preds.extend(output["predictions"])
                all_targets.extend(output["targets"])
            
            # Compute metrics
            wer = jiwer.wer(all_targets, all_preds)
            self.log("test_wer", wer, sync_dist=True)
        else:
            # Aggregate perplexity results
            avg_loss = sum(item["loss"] for item in self._test_outputs) / len(self._test_outputs)
            avg_perplexity = sum(item["perplexity"] for item in self._test_outputs) / len(self._test_outputs)
            
            self.log("test_avg_loss", avg_loss, sync_dist=True)
            self.log("test_avg_perplexity", avg_perplexity, sync_dist=True)
        self._test_outputs = []


    def configure_optimizers(self):
        # Collect parameters for visual encoder and transformer decoder
        if self.hparams.fusion_mode == "asr_only":
            optimizer = torch.optim.Adam(
                self.llm_params,
                lr=self.hparams.lora_lr,
                weight_decay=self.hparams.weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.max_steps
            )
            return [{'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}]
        
        if self.visual_only_training:
            optimizer = torch.optim.Adam(
                self.visual_params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=self.hparams.max_steps
            )
            return [{'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}]


        # Create optimizer for LoRA parameters
        lora_optimizer = torch.optim.Adam(
            self.llm_params,
            lr=self.hparams.lora_lr,
            weight_decay=self.hparams.weight_decay
        )
        # Create warmup scheduler for LoRA optimizer
        lora_scheduler = get_linear_schedule_with_warmup(
            lora_optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.max_steps
        )
        visual_optimizer = torch.optim.Adam(
            self.visual_params,
            lr=self.hparams.learning_rate,  # e.g. 1e-4
            weight_decay=self.hparams.weight_decay
        )
        visual_scheduler = get_linear_schedule_with_warmup(
            visual_optimizer,
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

