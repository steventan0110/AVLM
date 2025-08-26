# pre-train SpiritLM with LRS3 dataset to integrate visual modality

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer as pl_Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from transformers import HfArgumentParser, LlamaTokenizer
from src.models.avlm import AVLMModel
from src.data_utils.pretrain_data_util import AVLMPretrainDataModule
logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision('medium')

@dataclass
class ModelArguments:
    # General parameters
    mode: Optional[str] = field(default="train")  # train or eval
    model_path: Optional[str] = field(default=None)
    ckpt_path: Optional[str] = field(default=None)
    data_path: Optional[str] = field(default="./data")
    output_dir: Optional[str] = field(default="./checkpoints")
    run_name: Optional[str] = field(default="emotion_predictor_run")
    seed: Optional[int] = field(default=42)
    num_workers: Optional[int] = field(default=4)
    num_gpus: Optional[int] = field(default=1)

    # Training parameters
    max_epochs: Optional[int] = field(default=50)
    batch_size: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-4)
    lora_lr: Optional[float] = field(default=1e-4)
    weight_decay: Optional[float] = field(default=0.01)
    log_freq: Optional[int] = field(default=50)
    eval_and_save_every: Optional[int] = field(default=500)  # eval and save checkpoint every n train steps

    # Model parameters specific to audiovisual emotion prediction
    audio_feature_dim: Optional[int] = field(default=128)
    video_feature_dim: Optional[int] = field(default=18*32*32)
    temperature: Optional[float] = field(default=0.07)
    num_emotions: Optional[int] = field(default=5)
    n_layers: Optional[int] = field(default=6)
    d_model: Optional[int] = field(default=4096)
    n_head: Optional[int] = field(default=8)
    n_residual_layers: Optional[int] = field(default=3)
    num_warmup_steps: Optional[int] = field(default=4000)
    max_steps: Optional[int] = field(default=200000)
    grad_accum_every: Optional[int] = field(default=1)

    # other settings for ablation
    use_ckpt_path: Optional[str] = field(default=None)
    fusion_mode: Optional[str] = field(default="x-attn")
    resume_from: Optional[str] = field(default=None)
    feature_type: Optional[str] = field(default="vit")
    drop_audio_ratio: Optional[float] = field(default=0.0)
    test_drop_ratio: Optional[float] = field(default=0.0)
    expressive_mode: Optional[bool] = field(default=False)

def find_best_checkpoint(ckpt_dir):
    """
    Finds the checkpoint file with the lowest validation loss in the given directory.
    
    Args:
        ckpt_dir (str): Path to the checkpoint directory.
    
    Returns:
        str: Path to the checkpoint with the lowest validation loss.
    """
    assert os.path.isdir(ckpt_dir), f"Checkpoint directory {ckpt_dir} does not exist."
    
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
    
    # Regular expression to extract val_loss from filenames
    # pattern = re.compile(r"val_loss=([\d.]+)")
    pattern = re.compile(r"val_loss=([\d]+\.[\d]+)")
    
    best_ckpt = None
    lowest_loss = float("inf")
    
    for ckpt in ckpt_files:
        match = pattern.search(ckpt)
        if match:
            val_loss = float(match.group(1))  # Extract loss and convert to float
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                best_ckpt = ckpt

    if best_ckpt is None:
        raise ValueError(f"No valid checkpoint files found in {ckpt_dir}")
    
    return os.path.join(ckpt_dir, best_ckpt)

def init_model(args, ckpt_path=None, tokenizer=None):
    if ckpt_path is not None:  
        model = AVLMModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            hparams=vars(args),
            tokenizer=tokenizer,
            map_location="cpu" if args.num_gpus > 1 else None,
            strict=False
        )
    else:
        model = AVLMModel(hparams=vars(args), tokenizer=tokenizer)
    return model



class VisualSpiritLMTrainer:
    def __init__(self, args: ModelArguments) -> None:
        self.args = args
        pl.seed_everything(args.seed)
        os.makedirs(args.output_dir, exist_ok=True)

        # Set up the data module to load audiovisual emotion data
        spiritlm_checkpoints_dir = os.environ.get('SPIRITLM_CHECKPOINTS_DIR', './checkpoints/spiritlm')
        model_path = os.path.join(spiritlm_checkpoints_dir, 'spiritlm_model/spirit-lm-expressive-7b')
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            add_bos_token=False,
            add_eos_token=False,
        )
        logging.info(f"UNK token id: {tokenizer.unk_token_id}")
        logging.info(f"BOS token id: {tokenizer.bos_token_id}")
        logging.info(f"EOS token id: {tokenizer.eos_token_id}")
        logging.info(f"Vocab size: {tokenizer.vocab_size}")
        # since pad token is unset for pre-trained model, we need to manually set it
        my_own_pad_token = "[Madeuptoken32767]"
        pad_token_id = tokenizer.get_vocab()[my_own_pad_token]
        tokenizer.pad_token_id = pad_token_id
        tokenizer.pad_token = my_own_pad_token
        logging.info(f"Custom PAD token id: {tokenizer.pad_token_id}")
        logging.info(f"Custom PAD token: {tokenizer.pad_token}")
        logging.info(f"Special tokens: {tokenizer.special_tokens_map}")

        assert args.feature_type == "smirk"
        self.data_module = AVLMPretrainDataModule(
            tokenizer=tokenizer,
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            drop_audio_ratio=args.drop_audio_ratio,
            fusion_mode=args.fusion_mode,
            test_drop_ratio=args.test_drop_ratio,
            expressive_mode=args.expressive_mode
        )
        best_ckpt_path = None
        if args.mode == "test":
            if args.use_ckpt_path is not None:
                best_ckpt_path = args.use_ckpt_path
            else:
                assert args.ckpt_path is not None, "Please provide a model checkpoint for evaluation"
                best_ckpt_path = find_best_checkpoint(args.ckpt_path)
        elif args.ckpt_path is not None:
            logging.info(f"Initing Modeling training from {args.ckpt_path}")
            best_ckpt_path = args.ckpt_path
      

        self.model = init_model(args, ckpt_path=best_ckpt_path, tokenizer=tokenizer)
        logging.info(f"Initialized model: {self.model}")

        # Set up callbacks
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val_loss",  # Make sure your Lightning module logs a "val_loss"
            mode="min",  # We want to minimize the validation loss
            # every_n_epochs=1,
            save_top_k=3,
            verbose=True,
            every_n_train_steps=args.eval_and_save_every * 4 // args.grad_accum_every,
            # save_on_train_epoch_end=False,  # Don't save on epoch end since we're using steps
            filename="{step}-{val_loss:.4f}"
        )

        self.lr_monitor = LearningRateMonitor(logging_interval="step")

        # Set up logger (using Weights & Biases unless in debug mode)
        if args.run_name.startswith("debug") or args.mode == "test":
            self.logger = None
        else:
            random_string = str(int(time.time()))
            self.logger = WandbLogger(
                project="AVSR",
                name=f"{args.run_name}-{random_string}",
                save_dir=args.output_dir,
            )

        # Use DDP strategy if using more than one GPU
        strategy = "auto"
        if args.num_gpus > 1:
            logging.info(f"Using DDP strategy with {args.num_gpus} GPUs")
            strategy = DDPStrategy(find_unused_parameters=True)

        # Instantiate the Lightning Trainer with validation only every 10 epochs.
        self.trainer = pl_Trainer(
            default_root_dir=args.output_dir,
            max_epochs=args.max_epochs,
            log_every_n_steps=args.log_freq,
            callbacks=[self.checkpoint_callback, self.lr_monitor],
            devices=args.num_gpus if args.num_gpus > 0 else None,
            accelerator="gpu" if args.num_gpus > 0 else "cpu",
            logger=self.logger,
            strategy=strategy,
            val_check_interval=args.eval_and_save_every,  # Adjust for gradient accumulation
            # num_sanity_val_steps=-1,
            # num_sanity_val_steps=0
        )
        

    def train(self):
        if self.args.resume_from is not None:
            self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path=self.args.resume_from)
        else:
            self.trainer.fit(self.model, datamodule=self.data_module)

    def test(self):
        self.trainer.test(self.model, datamodule=self.data_module)


if __name__ == "__main__":
    # Parse command-line arguments into a ModelArguments dataclass
    args = HfArgumentParser(ModelArguments).parse_args_into_dataclasses()[0]
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Running with args: {args}")
    trainer = VisualSpiritLMTrainer(args)
    if args.mode == "train":
        trainer.train()
    else:
        trainer.test()