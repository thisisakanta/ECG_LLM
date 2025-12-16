export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=8 
TOTAL_BATCH_SIZE=8 # 144
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training

    CUDA_VISIBLE_DEVICES=0 python /home/user/Downloads/2005047/MEIT/ECG_LLMs/eval_ecgllm_test.py \
    --model_name_or_path huggyllama/llama-7b\
    --dataset_name /home/user/Downloads/2005047/MEIT/ECG_LLMs/ptbxl/ptbxl_ecg_train.jsonl \
    --save_dir /home/user/Downloads/2005047/MEIT/ECG_LLMs/eval_save_dir_llama1 \
    --use_slow_tokenizer \
    --eval_batch_size 8 \
    --load_in_8bit \
    --train_or_test test \
    --ecg_model_ckpt_path /home/user/Downloads/2005047/MEIT/ECG_LLMs/save_dir_llama1/ecg_model.pth\
    --with_tracking \
    --report_to tensorboard \
    --output_dir /home/user/Downloads/2005047/MEIT/ECG_LLMs/eval_save_dir_llama1_infer \
    

    


    
    # &&

# python open_instruct/merge_lora.py \
#     --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_lora/ \
#     --output_dir output/tulu_v2_${MODEL_SIZE}_lora_merged/ \
#     --save_tokenizer
