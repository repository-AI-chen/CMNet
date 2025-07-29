# -*- coding: utf-8 -*-
import os
import sys
import csv
import argparse
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument('--image_path', default=r'./openimage_v6/segmentation_validation/data', metavar="DIR",
                        help='Original images path')
    parser.add_argument('--current_checkpoint_bin_path', default=r'./save_results_real_bistream_temp/results_coco', metavar="DIR",
                        help='Bin path')
    return parser


args = get_parser().parse_args()

current_checkpoint_bin_path = args.current_checkpoint_bin_path
all_files = os.listdir(current_checkpoint_bin_path)
folders = []
for current_file in all_files:
    if not current_file.endswith('.txt'):
        folders.append(current_file)

for current_checkpoint_bin_folder in folders:
    bin_path = current_checkpoint_bin_path + '/' + str(current_checkpoint_bin_folder)

    bin_list = os.listdir(bin_path)

    img_path = args.image_path
    img_list = os.listdir(img_path)

    count = 0
    pixel_total = 0
    bit_total = 0
    for bin in bin_list:
        count += 1
        name = bin.split('.')[0]
        img_id = name + '.jpg'
        img_name = os.path.join(img_path, img_id)
        img = Image.open(img_name)
        W, H = img.size
        pixel = W * H
        bit = os.path.getsize(os.path.join(bin_path, bin))
        pixel_total += pixel
        bit_total += bit
    bpp_rate = bit_total * 8 / pixel_total
    print("********************checkpoint:{}*************************".format(current_checkpoint_bin_folder))
    print("Calculate {} images:\n\ttotal pixels: {}\n\ttotal bin size: {}\n\tbpp rate: {}"
        .format(count, pixel_total, bit_total, bpp_rate))
    print("You can find details in output folder.\nDone!")
    print("***********************************************************************")




