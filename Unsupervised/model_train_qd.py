# -*- coding: utf-8 -*-
# @Time    : 2024/6/22 17:10
# @Author  : Jay
# @File    : model_train_qd.py
# @Project: QD_detection
# optimal_k, optimal_d = 0.2, 10.5
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


# 读取图片并进行处理
def process_image(image_path, k, d):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}. Please check the path and file format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获取红色通道(QD_all)/绿色通道(FITC)
    red_channel = image_rgb[:, :, 0]
    # 初始化荧光计数矩阵
    fluorescence_counts = np.ones_like(red_channel, dtype=np.float32)
    # 遍历每个像素并计算荧光标记数
    mask = red_channel > k
    fluorescence_counts[mask] = red_channel[mask] / d + k

    return np.sum(fluorescence_counts / (2048 ** 2))


# 定义误差函数，最小化每对图片荧光数值差异的方差
def error_function(params):
    k, d = params
    calculated_fluorescence_values = np.array([process_image(path + file_name, k, d) for file_name in image_files])
    # 使用线性回归拟合
    model = LinearRegression()
    x = np.array([np.log10(10), np.log10(15), np.log10(20), np.log10(28), np.log10(58), np.log10(400), np.log10(3000), np.log10(30000), np.log10(300000), np.log10(3000000)]).reshape(-1, 1)
    y = calculated_fluorescence_values
    model.fit(x, y)
    predicted_values = model.predict(x)
    # 返回均方误差
    return np.mean((predicted_values - y) ** 2)


if __name__ == '__main__':
    path = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\train'
    image_files = [r'\10.bmp', r'\15.bmp', r'\20.bmp', r'\28.bmp', r'\58.bmp', r'\400.bmp', r'\3000.bmp', r'\30000.bmp', r'\300000.bmp', r'\3000000.bmp']

    # 优化阈值 k
    result = minimize(error_function, x0=np.array([1.0, 0.2]), bounds=[(0, 255), (0, 20)])
    optimal_k, optimal_d = result.x

    # 打印最佳阈值 k 和除数 d
    print(f"Optimal threshold k: {optimal_k}")
    print(f"Optimal divisor d: {optimal_d}")

    # 计算最佳阈值下每张图片的荧光数值
    calculated_fluorescence_values = [process_image(path + file_name, optimal_k, optimal_d) for file_name in
                                      image_files]

    # 打印每张图片的荧光数值
    for i, value in enumerate(calculated_fluorescence_values):
        print(f"Image {i + 1}: {value}")

    # 绘制结果
    plt.scatter(
        np.array([np.log10(10), np.log10(15), np.log10(20), np.log10(28), np.log10(58), np.log10(400), np.log10(3000), np.log10(30000), np.log10(300000), np.log10(3000000)]).reshape(-1, 1),
        calculated_fluorescence_values, color='blue',
        label='Calculated')
    plt.plot(
        np.array([np.log10(10), np.log10(15), np.log10(20), np.log10(28), np.log10(58), np.log10(400), np.log10(3000), np.log10(30000), np.log10(300000), np.log10(3000000)]).reshape(-1, 1),
        calculated_fluorescence_values, color='red',
        label='Fitted Line')
    plt.xlabel('Log')
    plt.ylabel('Fluorescence Value')
    plt.legend()
    plt.title('Calculated Fluorescence Values for Each Image')
    plt.show()



