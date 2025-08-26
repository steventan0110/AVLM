# AVLM Speech Only Evaluation Script

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=1e-4
lora_lr=3e-5
batch_size=12
grad_accum_every=4

drop_audio_ratio=0.0
fusion_mode="asr_only"
run_name="avlm_speech_only_pretrain_drop_${drop_audio_ratio}"
output_dir=$ROOT/ckpt/lrs/$run_name

test_drop=0.7
export PYTHONPATH=$REPO

export CUDA_VISIBLE_DEVICES=3; python $REPO/src/task/train_avlm.py \
    --num_gpus 1 \
    --fusion_mode $fusion_mode --drop_audio_ratio $drop_audio_ratio \
    --test_drop_ratio $test_drop \
    --run_name $run_name --n_layers 6 \
    --ckpt_path $output_dir --output_dir $output_dir \
    --data_path $LRS3SMIRK_LARGE --feature_type smirk \
    --mode "test" \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --eval_and_save_every 500 \
    --num_warmup_steps 0