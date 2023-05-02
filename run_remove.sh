python remove_OAOA.py \
    --input_img ./example/remove-anything/17331682166557_.pic.jpg \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama_init_bak/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama \