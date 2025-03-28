#!/bin/bash

models=(
    "meta-llama/Llama-3.2-3B-Instruct"
    "/home/YiWang/LLaMA-Factory/saves/llama-3.2-3B-dpo-selfdata-v1"

    "meta-llama/Meta-Llama-3-8B-Instruct"
    "/home/YiWang/LLaMA-Factory/saves/Llama-3-8B-Instruct/full/llama-3-8b-dpo"
    /home/YiWang/LLaMA-Factory/saves/Llama-3-8B-Instruct/full/llama-3-8b-simpo
    "declare-lab/trustalign_llama3_8b"

    "meta-llama/Llama-3.1-8B-Instruct"
    "WYJLUAI/llama3.1-8b-dpo"
    "/home/YiWang/LLaMA-Factory/saves/Llama-3.1-8B-Instruct/full/llama-3.1-8b-simpo"

    "Qwen/Qwen2.5-7B-Instruct"
    "/home/YiWang/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/Qwen2.5-7b-instruct-selfdata-dpo"
    "declare-lab/trustalign_qwen2.5_7b"
    
    "nvidia/Llama3-ChatQA-1.5-8B"
    "nvidia/Llama3-ChatQA-2-8B"
    
    "/home/YiWang/DFO-script/save_models/context-dpo-llama3-8b-instruct"
    "/home/YiWang/DFO-script/save_models/context-dpo-llama3.1-8b-instruct"
)
models=("meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "/home/YiWang/DFO-script/save_models/context-dpo-llama3-8b-instruct" "/home/YiWang/DFO-script/save_models/context-dpo-qwen2.5-7b-instruct" "/data/DFO/saves/test_llama3_tang" "/data/DFO/saves/test_qwen_model_tang")
datasets=( "eli5" )
# "nqopen" "nqswap" "xsum" "triviaqa" "memo-trap" "eli5"

# Loop through each model and dataset combination
for model in "${models[@]}"; do
    if [[ "$model" == *"qwen"* || "$model" == *"Qwen"* ]]; then
        export CUDA_VISIBLE_DEVICES=4,5,6,7
        gpu_num=4
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        gpu_num=8
    fi
    for dataset in "${datasets[@]}"; do
        echo "Running model $model on dataset $dataset"
        python main.py --model-id "$model" --eval-dataset "$dataset" --gpu_num $gpu_num
    done
done

# model='/home/XiaqiangTang/code/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/DFO_Qwen2.5-7B-Instruct_2'
# datasets=( "nqopen" "nqswap" "memo-trap" )

# for dataset in "${datasets[@]}"; do
#     echo "Running model $model on dataset $dataset"
#     python main.py --model-id "$model" --eval-dataset "$dataset" --gpu_num 4
# done