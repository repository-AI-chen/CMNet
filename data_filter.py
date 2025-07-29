import cv2
import os
 
def get_image_paths(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
 
# 使用例子
folder_path = './data'  # 替换为您的文件夹路径
image_paths = get_image_paths(folder_path)
number = 0
for index in range(len(image_paths)):
    img = cv2.imread(image_paths[index])
    H, W, C = img.shape
    if H<512 or W<512:
        os.remove(image_paths[index])
        number += 1
print(number)
