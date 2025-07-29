# Copyright (c) 2020-2021 Nokia Corporation and/or its subsidiary(-ies). All rights reserved.
# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

import struct
import numpy as np
import torch, torchvision
import torch.nn.functional as F
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2
import os

from detectron2.utils.logger import setup_logger
import struct

setup_logger()

import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling.meta_arch import GeneralizedRCNN

# import sys
import oid_mask_encoding

# import matplotlib.pyplot as plt

import argparse


class PLayerPredictor():
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, features, do_postprocess: bool = True, ):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            batched_inputs = [inputs]
            images = self.model.preprocess_image(batched_inputs)

            # x = torch.as_tensor([features]).cuda().float()
            if self.model.proposal_generator is not None:
                proposals, _ = self.model.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.model.device) for x in batched_inputs]

            results, _ = self.model.roi_heads(images, features, proposals, None)

            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
            else:
                return results


parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=str, default='0', \
                    help='task id')
parser.add_argument('--image_dir', type=str, default='./', \
                    help='image directory')
parser.add_argument('--task', type=str, default='detection', \
                    help='task: detection or segmentation')
parser.add_argument('--input_file', type=str, default='input.lst', \
                    help='input file that contains a list of image file names')
parser.add_argument('--yuv_dir', type=str, default='.', \
                    help='directory that containes (reconstructed) stem feature maps')
parser.add_argument('--output_file', type=str, default='output_coco.txt', \
                    help='prediction output file in OpenImages format')
args = parser.parse_args()

if args.task == 'detection':
    # model_cfg_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    model_cfg_name = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    # model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
elif args.task == 'segmentation':
    model_cfg_name = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
else:
    assert False, print("Unrecognized task:", args.task)

# construct detectron2 model
print('constructing detectron model ...')

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
p_predictor = PLayerPredictor(cfg)

# prediciton output
output_fname = args.output_file

coco_classes_fname = './data/coco_classes.txt'
with open(coco_classes_fname, 'r') as f:
    coco_classes = f.read().splitlines()

of = open(output_fname, 'w')

# write header
if args.task == 'detection':
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax\n')
else:
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')

################################################3333
# The same min/max values are obtained from the partially different image sets.
if args.task=="segmentation":
    global_max = 28.397470474243164
    global_min = -26.426828384399414
elif args.task=="detection":
    global_max = 28.397470474243164
    global_min = -26.426828384399414

# iterate all (reconstructed) stem files
with open(args.input_file, 'r') as f:
    with open(f"yuvinfo_{args.task}.txt", 'r') as f_yuvinfo:
        for idx, yuv_info in enumerate(f_yuvinfo.readlines()):
            # 1. Load YUV / Dequantisation / unpacking
            # 1-1 Load YUV
            orig_yuv_fname, width, height = yuv_info.split(',')
            width = int(width)
            height = int(height)
            flattened_plane = []


            if args.task_id=="UNCOMPRESSED":
                yuv_fname = orig_yuv_fname.replace('_UNCOMPRESSED','')
            else:
                yuv_fname = os.path.splitext(orig_yuv_fname)[0] + "_{}".format(args.task_id.replace("qp", "q")) + '.yuv'

            yuv_full_fname = os.path.join(args.yuv_dir, yuv_fname)
            print(f'processing {idx}:{yuv_full_fname}...')

            # with open(yuv_full_fname, "rb") as f_yuv:
            #     bytes = f_yuv.read(2)
            #     while bytes:
            #         val = int.from_bytes(bytes, byteorder='little')
            #         flattened_plane.append(val)
            #         bytes = f_yuv.read(2)
            #     flattened_plane = np.array(flattened_plane)
            #     q_plane = flattened_plane.reshape(height, width)
            #
            # print("DEBUG1")
            # # 1-2 Dequantisation
            # bits = 10
            # steps = np.power(2, bits) - 1
            # scale = steps / (global_max - global_min)
            # dq_plane = q_plane / scale + global_min
            #
            # print("DEBUG2")
            # # 1-3 Unpacking
            # pyramid = {}
            # v2_h = int(height / 85 * 64)
            # v3_h = int(height / 85 * 80)
            # v4_h = int(height / 85 * 84)
            #
            # v2_blk = dq_plane[:v2_h, :]
            # v3_blk = dq_plane[v2_h:v3_h, :]
            # v4_blk = dq_plane[v3_h:v4_h, :]
            # v5_blk = dq_plane[v4_h:height, :]
            #
            # pyramid["p2"] = feature_slice(v2_blk, [v2_blk.shape[0] // 16, v2_blk.shape[1] // 16])
            # pyramid["p3"] = feature_slice(v3_blk, [v3_blk.shape[0] // 8, v3_blk.shape[1] // 32])
            # pyramid["p4"] = feature_slice(v4_blk, [v4_blk.shape[0] // 4, v4_blk.shape[1] // 64])
            # pyramid["p5"] = feature_slice(v5_blk, [v5_blk.shape[0] // 2, v5_blk.shape[1] // 128])
            #
            # pyramid["p2"] = torch.unsqueeze(pyramid["p2"], 0)
            # pyramid["p3"] = torch.unsqueeze(pyramid["p3"], 0)
            # pyramid["p4"] = torch.unsqueeze(pyramid["p4"], 0)
            # pyramid["p5"] = torch.unsqueeze(pyramid["p5"], 0)
            #
            # # print(pyramid["p2"])
            # # print(pyramid["p3"])
            # # print(pyramid["p4"])
            # # print(pyramid["p5"])
            #
            # pyramid["p6"] = F.max_pool2d(pyramid["p5"], kernel_size=1, stride=2, padding=0)

            print("DEBUG3")
            # 2. Task performance evaluation
            img_fname = os.path.join(args.image_dir, os.path.splitext(orig_yuv_fname)[0] + '.png')
            # print(img_fname)
            img = cv2.imread(img_fname)
            if img is None:
                img = cv2.imread(os.path.splitext(img_fname)[0] + '.jpg')
            assert img is not None, print(f'Image file not found: {img_fname}')
            # print(f'processing {img_fname}...')

            print("DEBUG4")
            outputs = p_predictor(img, pyramid)[0]

            stemId = os.path.splitext(os.path.basename(orig_yuv_fname))[0]
            classes = outputs['instances'].pred_classes.to('cpu').numpy()
            scores = outputs['instances'].scores.to('cpu').numpy()
            bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
            H, W = outputs['instances'].image_size
            # convert bboxes to 0-1
            # detectron: x1, y1, x2, y2 in pixels
            bboxes = bboxes / [W, H, W, H]
            # OpenImage output x1, x2, y1, y2 in percentage
            bboxes = bboxes[:, [0, 2, 1, 3]]

            if args.task == 'segmentation':
                masks = outputs['instances'].pred_masks.to('cpu').numpy()

            print("DEBUG5")
            for ii in range(len(classes)):
                coco_cnt_id = classes[ii]
                class_name = coco_classes[coco_cnt_id]

                if args.task == 'segmentation':
                    assert (masks[ii].shape[1] == W) and (masks[ii].shape[0] == H), \
                        print('Detected result does not match the input image size: ', stemId)

                rslt = [stemId, class_name, scores[ii]] + \
                       bboxes[ii].tolist()

                if args.task == 'segmentation':
                    rslt += \
                        [masks[ii].shape[1], masks[ii].shape[0], \
                         oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                o_line = ','.join(map(str, rslt))

                of.write(o_line + '\n')
            print("DEBUG6")

        of.close()







