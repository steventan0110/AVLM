import torch
import os
import jiwer
import json
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from src.models.utils import VisualEncoder2D, get_attention_mask
from src.data_utils.mm_data_module import AudioVisualBatch, AVQFormerBatch
from transformers import get_linear_schedule_with_warmup, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from src.exp.spiritlm.spiritlm.model.spiritlm_model import Spiritlm
from src.exp.spiritlm.spiritlm.speech_tokenizer import spiritlm_expressive
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from src.models.qformer_utils import PerceiverResampler
from einops import repeat
from typing import Optional
eps=1e-12	# to prevent dividing by zero


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

@dataclass
class QFormerBatchItem:
    input_tokens: torch.Tensor
    input_mask: torch.Tensor
    target_tokens: torch.Tensor
    target_mask: torch.Tensor
    visual_feature: torch.Tensor
    visual_mask: torch.Tensor
    query_mask: torch.Tensor


class TextOnlyLogitsProcessor:
    def __call__(self, input_ids, scores):
        scores[:, 32000:] = float('-inf')  # mask non-text tokens
        return scores
    
            
def normalize_feature(feature, dim=-1, eps=1e-8):
    """
    Normalize feature values to the range [0, 1] along specified dimension.
    
    Args:
        feature (torch.Tensor): The input feature tensor to normalize
        dim (int): Dimension along which to normalize (default: -1)
        eps (float): Small epsilon value to prevent division by zero
    
    Returns:
        torch.Tensor: Normalized feature with values between 0 and 1
    """
    feature_min = feature.min(dim=dim, keepdim=True)[0]
    feature_max = feature.max(dim=dim, keepdim=True)[0]
    return (feature - feature_min) / (feature_max - feature_min + eps)




class MouthContourCNNEncoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=128):
        super(MouthContourCNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),  # (B, 32, N)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),              # (B, 64, N)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),             # (B, 128, N)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)                                   # (B, 128, 1)
        )
        
        self.fc = nn.Linear(128, latent_dim)  # Final latent projection

    def forward(self, x):
        # x: (B, T, 38) -> (B, T, 19, 2) -> (B*T, 19, 2)
        B, T, _ = x.shape
        x = x.view(x.shape[0], -1, 19, 2)
        x = x.view(-1, 19, 2).permute(0, 2, 1) # (B*T, 2, 19)
        x = self.encoder(x)  # (B*T, 128, 1)
        x = x.squeeze(-1)    # (B*T, 128)
        x = self.fc(x)       # (B*T, latent_dim)
        return x.view(B, T, -1)
    


class LandmarkCoefEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        landmark_dim = 19 * 2 # landmark feature dim
        coef_dim = 257 # coef feature dim

        self.landmark_encoder = MouthContourCNNEncoder()
        self.coef_fc = nn.Sequential(
            nn.Linear(coef_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, landmark_feature, coef_feature=None):
        if coef_feature is not None:
            coef_feature = normalize_feature(coef_feature)
            coef_embed = self.coef_fc(coef_feature)
            return coef_embed
        
        landmark_embed = self.landmark_encoder(landmark_feature)
        return landmark_embed
    
class SMIRKFeatureEncoder(nn.Module):
    def __init__(self, jaw_dim=3, expression_dim=50, output_dim=256):
        super().__init__()
        # 1D CNN for jaw features
        self.jaw_norm = nn.Sequential(
            nn.Linear(jaw_dim, jaw_dim),
            nn.ReLU(),
        )
        self.jaw_encoder = nn.Sequential(
            nn.Conv1d(jaw_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
        )
        
        # 1D CNN for expression features
        self.expression_norm = nn.Sequential(
            nn.Linear(expression_dim, expression_dim),
            nn.ReLU(),
        )
        self.expression_encoder = nn.Sequential(
            nn.Conv1d(expression_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
        )
        
        # Fusion layer to combine the features
        self.fuse_encoder = nn.Sequential(
            nn.Linear(32 + 128, 256),  # Combine jaw and expression features
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, jaw_feature, expression_feature):
        # Reshape for 1D convolution: [B, T, C] -> [B, C, T]
        B, T, _ = jaw_feature.shape
        jaw_feature = self.jaw_norm(jaw_feature)
        expression_feature = self.expression_norm(expression_feature)
        jaw_reshaped = jaw_feature.permute(0, 2, 1) # B, 3, T
        jaw_embed = self.jaw_encoder(jaw_reshaped).permute(0, 2, 1) # B, T, 32
        
        expression_reshaped = expression_feature.permute(0, 2, 1) # B, 50, T
        expression_embed = self.expression_encoder(expression_reshaped).permute(0, 2, 1) # B, T, 96

        # Combine features
        combined = torch.cat([jaw_embed, expression_embed], dim=-1)
        fuse_embed = self.fuse_encoder(combined)
        return fuse_embed



class MMAVLM(pl.LightningModule):
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
        self.visual_norm = nn.LayerNorm(128, dtype=torch.bfloat16)
        for param in self.visual_norm.parameters():
            self.trainable_fusion_params.append(param)
        
        if self.fusion_mode == "x-attn":
            self.max_latents = 512
            self.visual_encoder = LandmarkCoefEmbedder()
            self.speech_norm = nn.LayerNorm(d_model, dtype=torch.bfloat16)
            # self.context_proj = nn.Linear(d_model + 128, d_model, dtype=torch.bfloat16)
          
            for param in self.speech_norm.parameters():
                self.trainable_fusion_params.append(param)
            self.perceiver = PerceiverResampler(
                max_latents=self.max_latents,
                dim=d_model,
                depth=self.hparams.n_layers,
                dim_context=d_model+128,
                heads=32,
                dim_head=128
            ).to(torch.bfloat16)
            for param in self.perceiver.parameters():
                self.trainable_fusion_params.append(param)


        elif self.fusion_mode == "concate":
            self.visual_encoder = SMIRKFeatureEncoder()
            self.fusion_layer = nn.Sequential(
                nn.Linear(4096 + 256, 2048),
                nn.ReLU(),
                nn.Linear(2048, d_model)
            ).to(torch.bfloat16)

            for param in self.fusion_layer.parameters():
                self.trainable_fusion_params.append(param)
        elif self.fusion_mode == "qformer":
            self.max_latents = 128
            self.visual_encoder = SMIRKFeatureEncoder()
            self.perceiver = PerceiverResampler(
                max_latents=self.max_latents,
                dim=d_model,
                depth=self.hparams.n_layers,
                dim_context=256,
                heads=32,
                dim_head=128,
                use_pos_embed=False).to(torch.bfloat16)
            for param in self.perceiver.parameters():
                self.trainable_fusion_params.append(param)
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")

        for param in self.visual_encoder.parameters():
            self.trainable_fusion_params.append(param)

    

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
        visual_repr = self.visual_norm(visual_features)
        speech_repr = self.speech_norm(speech_repr)
        speech_repr[~speech_mask] = 0

        ######## Concatenate Speech and Visual Features  Along Time Dimension ########
        B, T_v, D_v = visual_features.shape
        fuse_feature = torch.cat([speech_repr, visual_repr], dim=2) # B, T_v, D_v + D_s
        attended_features = self.perceiver(fuse_feature, T_v, item.visual_mask)

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
        speech_drop_mask = speech_token == self.speech_mask_token
        if speech_drop_mask.sum() > 0:
            speech_embed[speech_drop_mask] = 0

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
        

    def compute_qformer_attention(self, item: QFormerBatchItem):
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)
        B, T, D = prompt_embed.shape 

        num_query_tokens = item.query_mask.sum(dim=-1)
        max_query = num_query_tokens.max()
        query_positions = self.create_index_matrix(item.query_mask, max_query)
        qformer_output = self.perceiver(item.visual_feature, max_query, mask=item.visual_mask) # B, Max_Query, D

        dummy_embed = prompt_embed.new_full((B, 1, D), 0)
        clone_embeds = torch.cat([prompt_embed.clone(), dummy_embed], dim=1)
        batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
        clone_embeds[batch_indices, query_positions] = qformer_output
        clone_embeds = clone_embeds[:, :-1]
        return clone_embeds


    def forward(self, item: BatchItem):
        # obtain embeddings
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(item.input_tokens)         
        # inputs_embeds = prompt_embed
        if self.fusion_mode == "concate":
            inputs_embeds = self.concate_fusion(item.speech_token, item.visual_feature, prompt_embed, item.visual_mask, item.speech_mask)
        elif self.fusion_mode == "x-attn":
            B, T, D = prompt_embed.shape 
            av_feature = self.compute_cross_attention(item.visual_feature, item)
            dummy_embeds = prompt_embed.new_full((B, 1, D), 0)
            clone_embeds = torch.concat([prompt_embed.clone(), dummy_embeds], dim=1)
            speech_token_max_len = av_feature.shape[1]
            speech_position = self.create_index_matrix(item.speech_mask, speech_token_max_len)
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1)
            clone_embeds[batch_indices, speech_position] = av_feature
            inputs_embeds = clone_embeds[:, :-1]
        elif self.fusion_mode == "qformer":
            inputs_embeds = self.compute_qformer_attention(item)
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
        for _, param in module.named_parameters():
            param_sum += param.sum().item()
        return param_sum

    def training_step(self, batch, batch_idx):
        item: BatchItem = self.prepare_batch(batch)
        # print learning rate
        logits = self(item)
        # print model weights to monitor
        # llm_param_sum = self.print_model_weights(self.spiritlm_model.model.model)
        # perceiver_param_sum = self.print_model_weights(self.visual_encoder)
        # context_proj_param_sum = self.print_model_weights(self.context_proj)
        # print(f"LLM params: {llm_param_sum}, Perceiver params: {perceiver_param_sum}, Context proj params: {context_proj_param_sum}")

        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(-1, V), item.target_tokens.view(-1))
        preds = torch.argmax(logits, dim=-1)
        # Only compute accuracy over positions where target is not -100 (masked positions)
        valid_mask = item.target_tokens != -100
        acc = (preds[valid_mask] == item.target_tokens[valid_mask]).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        

        visual_opt, lora_opt = self.optimizers()
        visual_scheduler, lora_scheduler = self.lr_schedulers()

        self.manual_backward(loss)
        # Only update on the last accumulation step
        if (batch_idx + 1) % self.hparams.grad_accum_every == 0:
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

            landmark_feature: torch.Tensor
            coef_feature: torch.Tensor
            visual_feature_len: torch.Tensor
            visual_mask: torch.Tensor

            text_token: Optional[torch.Tensor]
            text_token_len: Optional[torch.Tensor]
            text_mask: torch.Tensor

            # Prompt and prompt length
            prompt_token: torch.Tensor
            prompt_token_len: torch.Tensor
        """
        ### Prepare visual features #######
        if self.fusion_mode == "qformer":
            return self.prepare_qformer_batch(batch)
        visual_feature = self.visual_encoder(batch.jaw_feature, batch.expression_feature)
        # Get input and target sequences using shift-by-1 strategy
        input_tokens = batch.prompt_token[:, :-1]  # Remove last token for input
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
            speech_token=batch.speech_token,
            speech_mask=batch.speech_mask[:, :-1],
            visual_feature=visual_feature.to(torch.bfloat16),
            visual_mask=batch.visual_mask
        )

    def prepare_qformer_batch(self, batch: AVQFormerBatch):
        visual_feature = self.visual_encoder(batch.jaw_feature, batch.expression_feature)
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
        return QFormerBatchItem(
            input_tokens=input_tokens,
            input_mask=input_mask,
            target_tokens=labels,
            target_mask=target_mask,
            visual_feature=visual_feature.to(torch.bfloat16),
            visual_mask=batch.visual_mask,
            query_mask=batch.query_mask
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



        # print(input_ids)
        # print(attention_mask)
        # print(speech_mask)

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



    def test_qformer(self, item: QFormerBatchItem): 
        batch_size = item.input_tokens.shape[0]
        # first create the speech indices for fusion
        fused_embeds = self.compute_qformer_attention(item)
        text_start_positions = []
        max_end_pos = 0
        for i in range(batch_size):
            # Find start_of_text position
            text_start_pos = (item.input_tokens[i] == self.start_of_text_token).nonzero(as_tuple=True)[0]
            text_end_pos = text_start_pos[-1].item() + 2 # should be +1, +2 we expose it to the first text token
            text_start_positions.append(text_end_pos)
            max_end_pos = max(max_end_pos, text_end_pos)


        # Prepare padded input batch and attention mask as before
        left_pad_embeds = fused_embeds.new_full((batch_size, max_end_pos, fused_embeds.shape[-1]), 0)
        # Left pad each sequence in inputs_embeds
        for i in range(batch_size):
            end_pos = text_start_positions[i]
            left_pad_embeds[i, max_end_pos-end_pos:max_end_pos] = fused_embeds[i, :end_pos]

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

            num_target = legit_target_token.shape[0]
            pred_tokens = outputs[i]
            pre_token = item.input_tokens[i][text_start_positions[i]-1]
            
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
        item: BatchItem = self.prepare_batch(batch)
        if self.fusion_mode == "x-attn":
            self.test_attn(item)
        elif self.fusion_mode == "concate":
            self.test_concate(item)
        elif self.fusion_mode == "qformer":
            self.test_qformer(item)
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