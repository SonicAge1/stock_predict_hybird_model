import imageio
import os
import re

# 设置存储图片的目录
image_folder = './data'
gif_filename = './data/training_animation.gif'

# 获取所有图片文件名并按 epoch 排序
images = []

# 正则表达式用于匹配文件名中的 epoch 数字
epoch_pattern = re.compile(r'epoch_(\d+).png')

# 获取所有符合条件的文件名，并解析 epoch 数字
image_files = []
for file_name in os.listdir(image_folder):
    if file_name.endswith('.png'):
        match = epoch_pattern.search(file_name)
        if match:
            epoch_num = int(match.group(1))
            image_files.append((epoch_num, file_name))

# 按照 epoch 数字排序
image_files.sort(key=lambda x: x[0])

# 读取图片
for _, file_name in image_files:
    images.append(imageio.imread(os.path.join(image_folder, file_name)))

# 生成 GIF，duration 表示每帧的显示时间，单位为秒
imageio.mimsave(gif_filename, images, duration=0.5)
