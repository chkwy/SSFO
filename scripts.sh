#!/bin/bash
# Usage: bash scripts.sh [model_path] [model_name] [template]
# Description: Script to run the DFO training and evaluation pipeline
# Examples: 
# bash scripts.sh /home/KeyuHu/.cache/Llama-3.1-8B-Instruct test_llama3_1_hu llama3
# bash scripts.sh /home/KeyuHu/.cache/Meta-Llama-3-8B-Instruct test_llama3_hu llama3
# bash scripts.sh /home/KeyuHu/.cache/Qwen2.5-7B-Instruct test_qwen7B_model_hu qwen
# bash scripts.sh /home/KeyuHu/.cache/Qwen2.5-3B-Instruct test_qwen_model_hu qwen

set -e

model_path=${1:-Qwen/Qwen2.5-3B-Instruct}
model_name=${2:-test_model_tang}
template=${3:-llama3}
dataset_dir="${model_name}_DFO_data"

echo "Model path: $model_path"
echo "Model name: $model_name"
echo "Template: $template"
echo "Dataset dir: $dataset_dir"
echo "Check out the logs at logs/${model_name}_run.log"

echo "-------------------- Creating DFO data for model $model_name --------------------"
python create_dpo_data.py --model_name $model_name --model_path $model_path >> logs/${model_name}_run.log 2>&1
echo "DFO data created at $dataset_dir"

echo "-------------------- Training DFO model $model_name --------------------"
llamafactory-cli train \
    --stage dpo  \
    --do_train True \
    --model_name_or_path $model_path \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template $template \
    --flash_attn auto \
    --dataset_dir data \
    --dataset $dataset_dir \
    --cutoff_len 2048 \
    --learning_rate 1e-06 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 20 \
    --warmup_steps 10 \
    --packing False \
    --report_to none \
    --output_dir saves/$model_name \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid  \
    --deepspeed cache/ds_z3_config.json >> logs/${model_name}_run.log 2>&1

echo "Training completed. Model saved at saves/$model_name"

echo "-------------------- Starting evaluation --------------------"
cd eval

dfo_model="../saves/${model_name}"
datasets=( "nqopen" "nqswap" "memo-trap" "DuReader")
base_model_name=$(basename "$model_path")
base_dfo_model_name=$(basename "$dfo_model")

for dataset in "${datasets[@]}"; do
    echo "Evaluating $base_model_name on dataset $dataset"
    python main.py --model-id "$model_path" --eval-dataset "$dataset" --gpu_num 4 >> ../logs/${model_name}_run.log 2>&1

    echo "Evaluating $base_dfo_model_name on dataset $dataset"
    python main.py --model-id "$dfo_model" --eval-dataset "$dataset" --gpu_num 4 >> ../logs/${model_name}_run.log 2>&1
done


echo "-------------------- Evaluation results --------------------"

printf "%-20s | %-25s | %-25s\n" "Dataset" "Base model Subspan_EM" "DFO model Subspan_EM"
printf "%-20s-+-%-25s-+-%-25s\n" "$(printf '%.0s-' {1..20})" "$(printf '%.0s-' {1..25})" "$(printf '%.0s-' {1..25})"

for dataset in "${datasets[@]}"; do
    base_metric=$(cat outputs/${dataset}_${base_model_name}_metrics_final.json | \
                  tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                  awk '{printf "%.2f%%", $1*100}')

    dfo_metric=$(cat outputs/${dataset}_${base_dfo_model_name}_metrics_final.json | \
                 tr ',' '\n' | grep '"sub_Subspan_EM"' | cut -d':' -f2 | tr -d ' }' | \
                 awk '{printf "%.2f%%", $1*100}')

    printf "%-20s | %-25s | %-25s\n" "$dataset" "$base_metric" "$dfo_metric"
done
