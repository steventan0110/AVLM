# AVLM Emotion QFormer Generation Script

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=3e-4
lora_lr=5e-5
batch_size=5
grad_accum_every=8

fusion_mode="qformer"
run_name="avlm_iemocap_emo_${fusion_mode}"
output_dir=$ROOT/ckpt/lrs/$run_name

export PYTHONPATH=$REPO

# we fine-tune the avlm model on small lrs3 dataset
export CUDA_VISIBLE_DEVICES=0; python $REPO/src/task/avlm_iemocap_tune.py \
    --num_gpus 1 --test_gen True \
    --fusion_mode $fusion_mode \
    --force_emo_pred 4 \
    --run_name $run_name \
    --output_dir $output_dir \
    --ckpt_path $output_dir \
    --data_path $EMOWRITE --feature_type smirk \
    --mode "test" \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --num_warmup_steps 0