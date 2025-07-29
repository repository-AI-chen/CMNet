import os
import subprocess
import argparse

import torch
from future.moves import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description='运行coco2oid.py程序')
    parser.add_argument('-input_folder', type=str,
                        default=r'', help='包含输入文件的文件夹路径')
    parser.add_argument('-output_oi_folder', type=str,
                        default=r'', help='生成输出文件的文件夹路径')
    parser.add_argument('-output_metrics_folder', type=str,
                        default=r'', help='Output file with csv metrics.')
    parser.add_argument('-selected_classes', type=str,
                        default=r'', help='包含指定类别的文件路径')
    parser.add_argument('-input_annotations_boxes', type=str,
                        default=r'', help='File with groundtruth boxes annotations.')
    parser.add_argument('-input_annotations_labels', type=str,
                        default=r'', help='File with groundtruth labels annotations.')
    parser.add_argument('-input_class_labelmap', type=str,
                        default=r'', help='Open Images Challenge labelmap.')

    args = parser.parse_args(argv)
    return args

def run_coco2oid(coco_output_file, oid_output_file, selected_classes):
    """
  运行coco2oid.py程序。

  参数:
  coco_output_file (str): 输入文件路径。
  oid_output_file (str): 输出文件路径。
  selected_classes (str): 指定路径的某一个文件。
  """

    command = f"python cvt_detectron_coco_oid.py --coco_output_file {coco_output_file} --oid_output_file {oid_output_file} --selected_classes {selected_classes}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Command executed successfully.")
        print("Output:")
        print(result.stdout)
    else:
        print("Command failed with error:")
        print(result.stderr)

def run_oid_challenge_evaluation(input_annotations_boxes, input_annotations_labels,input_class_labelmap,input_predictions,output_metrics ):
    command = f"python oid_challenge_evaluation.py \
            --input_annotations_boxes   {input_annotations_boxes} \
            --input_annotations_labels  {input_annotations_labels} \
            --input_class_labelmap      {input_class_labelmap} \
            --input_predictions         {input_predictions} \
            --output_metrics            {output_metrics}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Command executed successfully.")
        print("Output:")
        print(result.stdout)
    else:
        print("Command failed with error:")
        print(result.stderr)


def main(argv):
    args = parse_args(argv)
    """
  主函数。

  参数:
  input_folder (str): 输入文件夹路径。
  output_folder (str): 输出文件夹路径。
  selected_classes (str): 指定路径的某一个文件。
  """
    if not os.path.exists(args.output_oi_folder):
        os.makedirs(args.output_oi_folder)

    for filename in os.listdir(args.input_folder):
        input_file = os.path.join(args.input_folder, filename)
        output_oi_file = os.path.join(args.output_oi_folder, filename)
        if os.path.exists(output_oi_file):
            print(f"Output oi file already exists! Skipping...")
            continue
        run_coco2oid(input_file, output_oi_file, args.selected_classes)

        output_result_file = os.path.join(args.output_metrics_folder, filename)
        if os.path.exists(output_result_file):
            print(f"Output result file already exists! Skipping...")
            continue
        run_oid_challenge_evaluation(args.input_annotations_boxes,args.input_annotations_labels, args.input_class_labelmap,output_oi_file, output_result_file)


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main(sys.argv[1:])
