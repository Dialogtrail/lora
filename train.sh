python3.10 finetune.py \
    --base_model '../llama-7b-hf' \
    --data_path '../data_kontextmannen_23000.json' \
    --output_dir './output_kontextmannen_23000' \
    --prompt_template_name 'eb' \
    --num_epochs 3 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --micro_batch_size 64 \
    --val_set_size 4000
#    --resume_from_checkpoint './output_kontextmannen4/checkpoint-600'
#    --wandb_project 'lora' \
#    --wandb_watch 'all'
#    --val_set_size 61
