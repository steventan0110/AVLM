# Script for training AVLM with QFormer to aggregate visual cues and infill the
# representation into speech tokens rather than putting in the front of the speech tokens

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=3e-4
lora_lr=3e-5
batch_size=8
grad_accum_every=4

drop_audio_ratio=0.5
fusion_mode="qformer_infill"
run_name="avlm_qformer_infilldrop_${drop_audio_ratio}"
output_dir=$ROOT/ckpt/lrs/$run_name
mkdir -p $output_dir

export PYTHONPATH=$REPO

speech_only_ckpt="YOUR_SPEECH_ONLY_CKPT"
    
python $REPO/src/task/train_avlm.py \
    --num_gpus 8 \
    --fusion_mode $fusion_mode --drop_audio_ratio $drop_audio_ratio \
    --run_name $run_name --n_layers 6 \
    --output_dir $output_dir \
    --ckpt_path $speech_only_ckpt \
    --data_path $LRS3SMIRK_LARGE --feature_type smirk \
    --mode "train" \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --eval_and_save_every 500 \
    --num_warmup_steps 1000