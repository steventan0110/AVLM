# AVLM QFormer Evaluation Script

# Get the directory where this script is located and source global config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../global.sh"

lr=1e-4
lora_lr=3e-5
batch_size=8
grad_accum_every=2

drop_audio_ratio=0.7
fusion_mode="qformer"
run_name="avlm_qformer_pretrain_drop_${drop_audio_ratio}"
output_dir=$ROOT/ckpt/lrs/$run_name


# eval qformer trained with pre-trained speech-only model, based on qformer arch
run_name="avlm_qformer_visual_only_drop_${drop_audio_ratio}"
output_dir=$ROOT/ckpt/lrs/$run_name

# # infill-based qformer
# fusion_mode="qformer_infill"
# run_name="avlm_qformer_infilldrop_${drop_audio_ratio}"
# output_dir=$ROOT/ckpt/lrs/$run_name


# # concate-based fusion
# fusion_mode="concate"
# run_name="avlm_concate_${drop_audio_ratio}"
# output_dir=$ROOT/ckpt/lrs/$run_name


echo "Evaluating $output_dir"

export PYTHONPATH=$REPO

test_drop=0.7

export CUDA_VISIBLE_DEVICES=0; python $REPO/src/task/train_avlm.py \
    --num_gpus 1 \
    --fusion_mode $fusion_mode --drop_audio_ratio $drop_audio_ratio \
    --run_name $run_name --n_layers 6 \
    --output_dir $output_dir --ckpt_path $output_dir \
    --data_path $LRS3SMIRK_LARGE --feature_type smirk \
    --mode "test" \
    --test_drop_ratio $test_drop \
    --max_epochs 100 \
    --batch_size $batch_size \
    --learning_rate $lr --lora_lr $lora_lr \
    --log_freq 10 \
    --grad_accum_every $grad_accum_every \
    --eval_and_save_every 500 \
    --num_warmup_steps 1000