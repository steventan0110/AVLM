import torch
import os
import jiwer
import json
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
from src.models.utils import VisualEncoder2D, get_attention_mask
from src.data_utils.pretrain_data_util import AudioVisualBatch

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report


eps=1e-12	# to prevent dividing by zero

EMO2ID = {
    "Neutral": 0,
    "Angry": 1,
    "Frustrated": 2,
    "Happy": 3,
    "Sad": 4
}

ID2EMO = {v: k for k, v in EMO2ID.items()}

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


class SpeechOnlyLogitsProcessor:
    def __init__(self):
        # Track the last token type that was generated
        self.last_token_type = None  # Can be "style", "pitch", "hubert", or None
        
        # Define token ranges
        self.style_min, self.style_max = 32567, 32666  # style token range
        self.pitch_min, self.pitch_max = 32503, 32566  # pitch token range
        self.hubert_min, self.hubert_max = 32002, 32502  # hubert token range

    def __call__(self, input_ids, scores):
        # First mask all non-speech tokens
        scores[:, :32002] = float('-inf') # 32000 is [Text], 32001 is [Speech], both we don't want to generate
        
        # Get current position in sequence
        current_pos = input_ids.shape[1] - 1
        if current_pos == -1:
            # first token, we only generate style token
            scores[:, self.pitch_min:self.pitch_max+1] = float('-inf')
            scores[:, self.hubert_min:self.hubert_max+1] = float('-inf')
            return scores
        
        # Check the most recently generated token type (if any)
        if current_pos >= 0:
            last_token = input_ids[0, current_pos].item()
            
            # Determine the type of the last token
            if self.style_min <= last_token <= self.style_max:
                self.last_token_type = "style"
            elif self.pitch_min <= last_token <= self.pitch_max:
                self.last_token_type = "pitch"
            elif self.hubert_min <= last_token <= self.hubert_max:
                self.last_token_type = "hubert"
        
        # Apply simple constraints to prevent consecutive style or pitch tokens
        if self.last_token_type == "style":
            # Don't allow another style token immediately after a style token
            scores[:, self.style_min:self.style_max+1] = float('-inf')
        
        if self.last_token_type == "pitch":
            # Don't allow another pitch token immediately after a pitch token
            scores[:, self.pitch_min:self.pitch_max+1] = float('-inf')
        
        # No restrictions on hubert tokens - they can appear consecutively
        
        return scores
    

class EmoClassifier(nn.Module):
    def __init__(self, d_model, num_emotions, hidden_dim=2048):
        super().__init__()
        
        # Two-layer linear network
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden_dim).to(torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        )
        
        # Final classification layer
        self.emo_classifier = nn.Linear(hidden_dim, num_emotions).to(torch.bfloat16)

    def forward(self, x):
        x = self.layers(x) # B, T, D
        x = torch.mean(x, dim=1)  # B, D
        return self.emo_classifier(x)
    

class JointEmoClassifier(nn.Module):
    def __init__(self, d_model, num_emotions, hidden_dim=2048):
        super().__init__()
        # Two-layer linear network
        self.visual_transform = nn.Sequential(
            nn.Linear(d_model, hidden_dim).to(torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        )
        self.speech_transform = nn.Sequential(
            nn.Linear(d_model, hidden_dim).to(torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        )
        # aggregate speech and visual cues for emotion prediction
        self.joint_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim).to(torch.bfloat16)
        )
        # Final classification layer
        self.emo_classifier = nn.Linear(hidden_dim, num_emotions).to(torch.bfloat16)

    def forward(self, visual_cue, speech_cue):
        visual_cue = self.visual_transform(visual_cue) # B, T, D
        speech_cue = self.speech_transform(speech_cue) # B, T, D
        x = torch.cat([visual_cue, speech_cue], dim=1) # B, T1+T2, D
        x = self.joint_transform(x) # B, T1+T2, D
        x = torch.mean(x, dim=1)  # B, D
        return self.emo_classifier(x)
    
    
class AVLMForEmoModel(pl.LightningModule):
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
        self.test_gen = self.hparams.test_gen
        self.force_emo_pred = self.hparams.force_emo_pred
        if self.force_emo_pred is not None:
            logging.info(f"Force Emo Pred: {self.force_emo_pred}")
        d_model = self.hparams.d_model
        spiritlm_path = os.path.join(os.environ['SPIRITLM_CHECKPOINTS_DIR'], "spiritlm_model", "spirit-lm-expressive-7b")

        if self.hparams.ckpt_path is not None: # loading from ckpt means we don't want to load pre-trained weights
            config = LlamaConfig.from_pretrained(spiritlm_path)
            spiritlm_model = LlamaForCausalLM(config=config).to(torch.bfloat16)
            # self.visual_only_training = True # add true here if we want to disable LLM tuning
        else:
            spiritlm_model = LlamaForCausalLM.from_pretrained(
                spiritlm_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)

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
        if self.hparams.fusion_mode == "speech_only":
            self.use_last_n_layer = -1
            logging.info("No visual encoder as ASR only")
            self.emo_classifier = EmoClassifier(d_model, len(EMO2ID)).to(torch.bfloat16)
            for param in self.emo_classifier.parameters():
                self.visual_params.append(param)

        elif self.hparams.fusion_mode == "qformer":
            self.use_last_n_layer = -4
            logging.info("QFormer Mode Used")
            self.max_latents = 128
            self.visual_encoder = SMIRKFeatureEncoder()
            self.perceiver = PerceiverResampler(
                max_latents=self.max_latents,
                dim=d_model,
                depth=self.hparams.n_layers,
                dim_context=256,
                heads=32,
                dim_head=128).to(torch.bfloat16)
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            for param in self.perceiver.parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            self.perceiver.eval()
            self.joint_emo_classifier = JointEmoClassifier(d_model, len(EMO2ID)).to(torch.bfloat16)
            for param in self.joint_emo_classifier.parameters():
                self.visual_params.append(param)
        else:
            raise ValueError(f"Invalid fusion mode: {self.hparams.fusion_mode}")


        # Loss function.
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self._val_outputs, self._test_outputs = [], []
        self.automatic_optimization = False

        self.emo_prompt = {}
        for emo in EMO2ID.keys():
            emo_prompt = f"\nEmotion: {emo}\nResponse:"
            emo_index = EMO2ID[emo]
            self.emo_prompt[emo_index] = self.tokenizer.encode(emo_prompt, add_special_tokens=False)


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
        prompt_embed = self.spiritlm_model.model.model.embed_tokens(input_tokens)
        B, T, D = prompt_embed.shape 
        target_mask = labels != -100

        # pool out pitch and style token position embeddings from the input
        # style token range: 32567 - 32666
        # pitch token range: 32503 - 32566
        # hubert token range: 32001 - 32502
        # Create mask for pitch tokens (32503-32566) and style tokens (32567-32666)
        pitch_style_mask = (input_tokens >= 32503) & (input_tokens <= 32666) & (~target_mask)

        if self.hparams.fusion_mode == "qformer":
            with torch.no_grad():
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
        else:
            # speech-only mode
            inputs_embeds = prompt_embed

        out: CausalLMOutputWithPast = self.spiritlm_model(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=labels,
            output_hidden_states=True
        )
        last_hidden_state = out.hidden_states[self.use_last_n_layer] # B, T, D
        all_emo_pred = []
        for i in range(B):
            speech_cue = last_hidden_state[i, pitch_style_mask[i]].unsqueeze(0)
            if self.hparams.fusion_mode == "speech_only":
                emo_pred = self.emo_classifier(speech_cue.detach()) # detach to prevent gradient flow
            else:
                visual_cue = last_hidden_state[i, item.query_mask[i]].unsqueeze(0)
                emo_pred = self.joint_emo_classifier(visual_cue.detach(), speech_cue.detach())
            all_emo_pred.append(emo_pred)
        all_emo_pred = torch.cat(all_emo_pred, dim=0)
        logits = out.logits # B, T, Vocab_size
        return logits, all_emo_pred
    

    @staticmethod
    def print_model_weights(module):
        param_sum = 0.0
        for _, param in module.named_parameters():
            param_sum += param.sum().item()
        return param_sum


    def training_step(self, batch: AudioVisualBatch, batch_idx):
        logits, emo_pred = self.forward(batch.input_token, batch.labels, batch)

        # llm_param_sum = self.print_model_weights(self.spiritlm_model.model.model)
        # perceiver_param_sum = self.print_model_weights(self.visual_encoder)
        # print(f"LLM params: {llm_param_sum}, Perceiver params: {perceiver_param_sum}")

        # Reshape logits to (B*T, V) and labels to (B*T) for cross entropy loss
        B, T, V = logits.shape

        target_mask = batch.labels != -100
        valid_logits = logits[target_mask]
        valid_labels = batch.labels[target_mask]

        loss = self.loss_fn(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=-1)
        acc = (preds == valid_labels).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        
        emo_loss = self.loss_fn(emo_pred, batch.emo_label)
        loss += emo_loss
        emo_acc = (torch.argmax(emo_pred, dim=-1) == batch.emo_label).float().mean()
        self.log("train_emo_acc", emo_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train_emo_loss", emo_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        self.manual_backward(loss)
        # Only update on the last accumulation step
      
        if (batch_idx + 1) % self.hparams.grad_accum_every == 0:
            self.clip_gradients(optimizer, 1.0, "norm") 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return loss


    def validation_step(self, batch: AudioVisualBatch, batch_idx):
        labels = batch.labels
        logits, emo_pred = self.forward(batch.input_token, labels, batch)

        B, T, V = logits.shape
        target_mask = labels != -100
        valid_logits = logits[target_mask]
        valid_labels = labels[target_mask]

        loss = self.loss_fn(valid_logits, valid_labels)
        emo_loss = self.loss_fn(emo_pred, batch.emo_label)
        loss += emo_loss
        preds = torch.argmax(valid_logits, dim=-1)
        acc = (preds == valid_labels).float().mean()
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        # compute perplexity
        perplexity = torch.exp(loss)
        self.log("val_ppl", perplexity, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        gt_emo = batch.emo_label
        pred_emo = torch.argmax(emo_pred, dim=-1)
        emo_acc = (pred_emo == gt_emo).float().mean()
        self.log("val_emo_acc", emo_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        return loss


    def _dialogue_gen(self, batch: AudioVisualBatch):
        assert batch.input_token.shape[0] == 1, "Only one sample per batch is supported for dialogue generation"
        _, emo_pred = self.forward(batch.input_token, batch.labels, batch)
        use_gt_emotion = False
        emo_pred = torch.argmax(emo_pred, dim=-1).item()
        emo_text = ID2EMO[emo_pred]
        gt_emo = batch.emo_label.item()
        gt_emo_text = ID2EMO[gt_emo]


        if use_gt_emotion:
            # we do not swap the provided emotion in the prompt with model prediction
            prefix_mask = batch.labels == -100 # this is only working when we have batch size 1
            prefix_len = prefix_mask.sum()
            prefix_tokens = batch.input_token[:, :prefix_len+1] # +1 for the [speech_token]
            prefix_text = self.tokenizer.decode(prefix_tokens[0], skip_special_tokens=False)
        else:
            # swap the provided emotion in the prompt with model prediction
            prefix_mask = batch.labels == -100 # this is only working when we have batch size 1
            prefix_len = prefix_mask.sum()
            prefix_tokens = batch.input_token[:, :prefix_len+1] # +1 for the [speech_token]
            
            original_emo_token = self.emo_prompt[gt_emo]

            # emo_pred = 0 # force prediction to always be neutral
            # emo_pred = 1 # force prediction to always be angry
            # emo_pred = 3 # force prediction to always be happy
            # emo_pred = 4 # force prediction to always be sad
            if self.force_emo_pred is not None:
                emo_pred = self.force_emo_pred
            updated_emo_token = self.emo_prompt[emo_pred]
            prefix_text = self.tokenizer.decode(prefix_tokens[0], skip_special_tokens=False)


            prefix_tokens_list = prefix_tokens[0].tolist()
            # Find the starting position of [29871, 13] in the prefix_tokens
            emo_start_idx = -1
            for i in range(len(prefix_tokens_list) - 1):
                if prefix_tokens_list[i] == 29871 and prefix_tokens_list[i+1] == 13:
                    emo_start_idx = i
                    break
            
            if emo_start_idx == -1:
                # Couldn't find emotion token sequence
                logging.warning("Could not find emotion token sequence pattern [29871, 13]")
                prefix_text = self.tokenizer.decode(prefix_tokens[0], skip_special_tokens=False)
            else:
                # Create a new token list by concatenating three parts:
                # 1. Tokens before emotion section
                # 2. The new emotion tokens
                # 3. Tokens after the emotion section
                
                # Find where the original emotion sequence ends
                emo_end_idx = emo_start_idx + len(original_emo_token)
                
                # Build new token list
                new_prefix_tokens_list = prefix_tokens_list[:emo_start_idx] + updated_emo_token + prefix_tokens_list[emo_end_idx:]
                
                # Convert back to tensor
                new_prefix_tokens = torch.tensor([new_prefix_tokens_list], device=batch.input_token.device)
 
                # Decode the new prefix tokens
                prefix_text = self.tokenizer.decode(new_prefix_tokens[0], skip_special_tokens=False)



        if self.hparams.fusion_mode == "qformer":
            prompt_embed = self.spiritlm_model.model.model.embed_tokens(prefix_tokens)
            visual_feature = self.visual_encoder(batch.jaw_feature, batch.expression_feature).to(torch.bfloat16)
            num_query_tokens = batch.query_mask.sum(dim=-1)
            max_query = num_query_tokens.max()
            query_positions = self.create_index_matrix(batch.query_mask, max_query)
            qformer_output = self.perceiver(visual_feature, max_query, mask=batch.visual_mask) # B, Max_Query, D
            prompt_embed[:, query_positions, :] = qformer_output
            prefix_embed = prompt_embed
            # prefix_embed = prompt_embed[:, :prefix_len+1, :]
        else:
            prefix_embed = self.spiritlm_model.model.model.embed_tokens(prefix_tokens)

        outputs = self.spiritlm_model.generate(
            inputs_embeds=prefix_embed,
            # early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            logits_processor=[SpeechOnlyLogitsProcessor()],
            generation_config=GenerationConfig(
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=300,
                do_sample=True,
            ),
        )
        pred_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Predicted Text: {pred_tokens}")
        cur_key = batch.keys[0]
        json_obj = {
            "prefix_text": prefix_text,
            "pred_text": pred_tokens,
            "gt_emo": gt_emo_text,
            "pred_emo": emo_text,
            "key": cur_key
        }

        self._test_outputs.append(json_obj)



    def test_step(self, batch: AudioVisualBatch, batch_idx):
        if self.test_gen:
            return self._dialogue_gen(batch)
        
        logits, emo_pred = self.forward(batch.input_token, batch.labels, batch)
        B, T, V = logits.shape
        gt_emo = batch.emo_label
        self._test_outputs.append({
            "emo_pred": emo_pred,
            "emo_label": gt_emo
        })


    def _dialogue_gen_end(self):
        save_dir = os.environ.get('GEN_OUTPUT_DIR', './output/generation')
        if self.force_emo_pred is not None:
            save_file = os.path.join(save_dir, f"{self.hparams.fusion_mode}_force_{self.force_emo_pred}.jsonl")
        else:
            save_file = os.path.join(save_dir, f"{self.hparams.fusion_mode}_ngt.jsonl")
        write_handle = open(save_file, "w")
        for json_obj in self._test_outputs:
            write_handle.write(json.dumps(json_obj) + "\n")
        write_handle.close()
        self._test_outputs = []



    def on_test_epoch_end(self):
        if self.test_gen:
            return self._dialogue_gen_end()
        
        # Gather predictions and targets for more detailed metrics
        all_preds = torch.cat([torch.argmax(out["emo_pred"], dim=-1) for out in self._test_outputs])
        all_targets = torch.cat([out["emo_label"] for out in self._test_outputs])
        # we group frustrated and angry as angry
        all_preds[all_preds == 2] = 1
        all_targets[all_targets == 2] = 1
        
        # Convert to numpy for sklearn metrics
        all_preds_np = all_preds.cpu().numpy()
        all_targets_np = all_targets.cpu().numpy()
        
        # Calculate additional metrics
            
        unweighted_average = accuracy_score(all_targets_np, all_preds_np)
        weighted_average = balanced_accuracy_score(all_targets_np, all_preds_np)
        macro_f1 = f1_score(all_targets_np, all_preds_np, average="macro")
        
        # Log metrics
        self.log("test_unweighted_acc", unweighted_average, sync_dist=True)
        self.log("test_weighted_acc", weighted_average  , sync_dist=True)
        self.log("test_macro_f1", macro_f1, sync_dist=True)
        
        # Log detailed classification report
        logging.info("Classification Report:")
        emo_list = ['Neutral', 'Angry', 'Happy', 'Sad']
        logging.info("\\n" + classification_report(all_targets_np, all_preds_np, 
                                   target_names=emo_list))
        
        self._test_outputs = []



    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.llm_params,
        #     lr=self.hparams.lora_lr,
        #     weight_decay=self.hparams.weight_decay
        # )
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
    
