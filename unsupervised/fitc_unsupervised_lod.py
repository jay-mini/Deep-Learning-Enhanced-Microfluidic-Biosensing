# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 15:33
# @Author  : Jay
# @File    : fitc_unsupervised_lod.py
# @Project: QD_detection
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams

config = {
    "font.size": 80,
    "mathtext.fontset": 'stix',
}
rcParams.update(config)


def process_image(image_path, k, d):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}. Please check the path and file format.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel = image_rgb[:, :, 1]  # 获取红色通道 / 绿色通道(FITC)
    fluorescence_counts = np.ones_like(red_channel, dtype=np.float32)
    mask = red_channel > 0
    fluorescence_counts[mask] = red_channel[mask] / d + k

    fluorescence_sum = np.sum(fluorescence_counts / (2048 * 2048))

    return fluorescence_sum


def process_images_in_directory(directory, k, d):
    files = os.listdir(directory)
    fluorescence_values = []
    image_names = []
    for filename in files:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            fluorescence_value = process_image(file_path, k, d)
            if fluorescence_value is not None:
                fluorescence_values.append(fluorescence_value)
                image_names.append(filename)
    return image_names, fluorescence_values


def get_fluorescence_statistics_for_background(background_directory, k, d):
    image_names, fluorescence_values = process_images_in_directory(background_directory, k, d)
    return image_names, fluorescence_values


if __name__ == '__main__':
    main_directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\FITC'
    background_directory = os.path.join(main_directory, 'background')
    optimal_k, optimal_d = 0.2, 10.5

    # 读取并处理 background 文件夹内的图片
    image_names, fluorescence_values = get_fluorescence_statistics_for_background(background_directory, optimal_k, optimal_d)

    # 保存数据到 Excel 文件，不保存 Actual Log Label
    data = {
        'Image Name': image_names,
        'Fluorescence Value': fluorescence_values
    }
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(background_directory, 'fluorescence_data_background.xlsx'), index=False)
