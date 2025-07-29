import cv2
import os
 
def get_image_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
 
# 使用例子
folder_path = 'D:\\doctoral_chen\\file\\code\\datasets\\openimage_v6\\validation\\data'  # 替换为您的文件夹路径
image_paths = get_image_paths(folder_path)
number = 0
for index in range(len(image_paths)):
    img = cv2.imread(image_paths[index])
    H, W, C = img.shape
    if H<512 or W<512:
        os.remove(image_paths[index])
        number += 1
# 下载的训练9000张图像里面有1110张是尺寸是小于512*512
# 下载的训练5000张图像里面有517张是尺寸是小于512*512
print(number)