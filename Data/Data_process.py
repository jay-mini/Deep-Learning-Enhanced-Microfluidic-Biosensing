# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 9:26
# @Author  : Jay
# @File    : Data_process.py
# @Project: main.py
# 对数据进行重命名处理
import os
import cv2
import numpy as np


# 批量重命名文件夹内的文件
def rename_files_in_directory(directory, new_prefix):
    files = os.listdir(directory)
    for i, filename in enumerate(files):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            new_name = f"{new_prefix}_{i + 1}.bmp"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")


if __name__ == '__main__':
    directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\FITC\background'
    new_prefix = 'image'

    # 批量重命名文件
    rename_files_in_directory(directory, new_prefix)
    print("数据处理完成！")
