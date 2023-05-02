CUDA_VISIBLE_DEVICES=6

python app.py \
      --lama_config ./lama/configs/prediction/default.yaml \
      --lama_ckpt ./pretrained_models/big-lama \
      --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth

