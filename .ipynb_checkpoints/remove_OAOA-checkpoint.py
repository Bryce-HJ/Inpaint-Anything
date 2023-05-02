import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import yaml
import logging
import copy

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo

# from sam_segment import predict_masks_with_sam
# from lama_inpaint import inpaint_img_with_lama

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )
    parser.add_argument('--draw_masks', action='store_true',
                   help="Draw and save the masks segmented by SAM")


if __name__ == "__main__":
    """Example usage:
    python remove_OAOA.py \
        --input_img FA_demo/FA1_dog.png \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama \
    """
    boxes = [[1260, 744, 1506, 1070], 
             [514, 723, 678, 843], 
             [1180, 733, 1234, 1070], 
             [1274, 885, 1312,952],
             [1210,876,1244,951],
             [1203, 740,1225, 790]]
        
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    sam_checkpoint = args.sam_ckpt
    model_type = args.sam_model_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "default"
    MAX_WIDTH = MAX_HEIGHT = 800
    THRESHOLD = 0.05

    print("Constructing SAM (VIT-Huge) models.....")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    print("SAM (VIT-Huge) models has been ready!")
    
    print('\n')
    print("Constructing LAMA models.....")
    # inpaint the masked image
    predict_config = OmegaConf.load(args.lama_config)
    predict_config.model.path = args.lama_ckpt
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)
    print("LAMA models has been ready!!")
    
    img = load_img_to_array(args.input_img)
    predictor.set_image(img)
    
#     point_coors = [[529, 578], [600, 200]]
        
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
        
    tmp_img = None
    for idx in range(len(boxes)):
        masks, _, _ = predictor.predict(box=np.array([boxes[idx]]), ) # np.array([1])
        masks = masks.astype(np.uint8) * 255
        if args.dilate_kernel_size is not None:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
            if args.draw_masks:
                for ii, m in enumerate(masks):
                    mask_p = out_dir / f"maskss_{idx}_{ii}.png"
                    save_array_to_img(m, mask_p)
        
        if tmp_img is not None and idx>0:
            img = torch.from_numpy(tmp_img).float().div(255.)
        else:
            img = torch.from_numpy(img).float().div(255.)
    
        # choose mask
        mask = torch.from_numpy(masks[-1]).float()

        batch = {}
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], 8)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], 8)
        batch = move_to_device(batch, device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = model(batch)
        cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        if idx >= 0:
            tmp_img = copy.deepcopy(cur_res)
            
    mask_p = out_dir / f"mask_{idx}.png"
    img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
    save_array_to_img(cur_res, img_inpainted_p)
    print("The processed image is saved in %s" % img_inpainted_p)
    