OUTPUT_DIR='/home/abhishek21441/VideoMAE1/Results/'
DATA_PATH='/home/abhishek21441/SSV2/20bn-something-something-v2-00'
MODEL_PATH='/home/abhishek21441/Model_VideoMAE/checkpoint.pth'
python run_class_finetuning.py \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 50 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --model vit_base_patch16_244
