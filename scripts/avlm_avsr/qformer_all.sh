# AVLM AVSR QFormer All Training Script

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=5e-5
lora_lr=1e-5
batch_size=8
grad_accum_every=4

drop_audio_ratio=0.0
fusion_mode="qformer_avsr"
run_name="avlm_all_${fusion_mode}_drop_${drop_audio_ratio}"
output_dir=$ROOT/ckpt/lrs/$run_name

mkdir -p $output_dir

export PYTHONPATH=$REPO

avlm_pretrain_ckpt="YOUR_AVLM_PRETRAIN_CKPT"

python $REPO/src/task/train_avlm.py \
    --num_gpus 8 \
    --ckpt_path $avlm_pretrain_ckpt \
    --fusion_mode $fusion_mode --drop_audio_ratio $drop_audio_ratio \
    --run_name $run_name --n_layers 6 \
    --output_dir $output_dir \
    --data_path $LRS3SMIRK_LARGE --feature_type smirk \
    --mode "train" \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --num_warmup_steps 0 \
    --eval_and_save_every 500