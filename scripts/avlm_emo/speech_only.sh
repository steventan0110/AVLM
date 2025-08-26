# AVLM Emotion Speech Only Training Script

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=5e-5
lora_lr=5e-5
batch_size=5
grad_accum_every=8

fusion_mode="speech_only"
run_name="avlm_iemocap_${fusion_mode}"
output_dir=$ROOT/ckpt/lrs/$run_name

mkdir -p $output_dir

export PYTHONPATH=$REPO


# we fine-tune the avlm model on small lrs3 dataset
python $REPO/src/task/avlm_iemocap_tune.py \
    --num_gpus 4 \
    --fusion_mode $fusion_mode \
    --run_name $run_name \
    --output_dir $output_dir \
    --data_path $EMOWRITE --feature_type smirk \
    --mode "train" \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --num_warmup_steps 0