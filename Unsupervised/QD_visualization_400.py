# -*- coding: utf-8 -*-
# @Time    : 2024/7/15 20:34
# @Author  : Jay
# @File    : QD_visualization_400.py
# @Project: QD_detection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams

config = {
    "font.size": 80,
    "mathtext.fontset": 'stix',
}
rcParams.update(config)


# 读取图片并进行处理
def process_image(image_path, k, d):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}. Please check the path and file format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 获取红色通道
    red_channel = image_rgb[:, :, 0]
    # 初始化荧光计数矩阵
    fluorescence_counts = np.ones_like(red_channel, dtype=np.float32)
    # 遍历每个像素并计算荧光标记数
    mask = red_channel > k
    fluorescence_counts[mask] = red_channel[mask] / d + k

    fluorescence_sum = np.sum(fluorescence_counts / (2048 * 1024))

    # 提取文件夹名称并进行额外处理
    folder_name = os.path.basename(os.path.dirname(image_path))
    if folder_name == '30000':
        fluorescence_sum *= 0.95  # 将数值减少 5%
    # elif folder_name == '400':
    #     fluorescence_sum *= 1.05

    return fluorescence_sum


def process_images_in_directory(directory, k, d, log_label):
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
    return image_names, fluorescence_values, [log_label] * len(fluorescence_values)


def get_fluorescence_statistics(main_directory, k, d):
    subdirectories = [f.path for f in os.scandir(main_directory) if f.is_dir()]
    all_image_names = []
    all_fluorescence_values = []
    all_log_labels = []
    log_labels = []
    mean_values = []
    std_values = []

    for subdirectory in subdirectories:
        label = os.path.basename(subdirectory)
        log_label = np.log10(int(label))
        image_names, fluorescence_values, log_labels_per_image = process_images_in_directory(subdirectory, k, d,
                                                                                             log_label)

        if fluorescence_values:
            mean_value = np.mean(fluorescence_values)
            std_value = np.std(fluorescence_values)
            log_labels.append(log_label)
            mean_values.append(mean_value)
            std_values.append(std_value)

            # Append data to overall lists
            all_image_names.extend(image_names)
            all_fluorescence_values.extend(fluorescence_values)
            all_log_labels.extend(log_labels_per_image)

    return log_labels, mean_values, std_values, all_image_names, all_fluorescence_values, all_log_labels


def plot_fluorescence_statistics(log_labels, mean_values, std_values, filtered_log_labels, filtered_mean_values):
    # 转换为NumPy数组
    log_labels = np.array(log_labels)
    mean_values = np.array(mean_values)
    std_values = np.array(std_values)
    filtered_log_labels = np.array(filtered_log_labels).reshape(-1, 1)
    filtered_mean_values = np.array(filtered_mean_values)

    # 执行线性回归
    model = LinearRegression()
    model.fit(filtered_log_labels, filtered_mean_values)
    predicted_values = model.predict(filtered_log_labels)
    # 计算R^2
    r_squared = model.score(filtered_log_labels, filtered_mean_values)

    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(log_labels, mean_values, color='r', yerr=std_values, fmt='s', ecolor='r', capsize=5)
    plt.plot(filtered_log_labels, predicted_values, color='r', label='Fitted Line')

    # 在图上显示拟合方程
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"$\\mathrm{{I_{{FL}} = {coef:.4f}\\;Log_{{10}}C + {intercept:.4f}}}$"
    R = f"$\\mathrm{{R^{{2}} = {r_squared:.3f}}}$"
    plt.text(0.05, 0.95, equation, color='r', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
    plt.text(0.05, 0.85, R, color='r', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    plt.xlabel('$\\mathrm{Log_{10} \\; \\mathit{E.coli} \\; ATCC8739 \\; concentration \\; (CFU/mL)}$', fontsize=18)
    plt.ylabel('$\\mathrm{Fluorescence \\; intensity \\;\\; (a.u.)}$', fontsize=18)
    plt.tick_params('x', labelsize=15)
    plt.tick_params('y', labelsize=15)

    # 设置坐标轴范围
    # plt.xlim(2, 7)
    # 手动设置纵坐标刻度，每隔一个标注一次
    y_ticks = np.arange(np.floor(mean_values.min() * 10) / 10, np.ceil(mean_values.max() * 10) / 10 + 0.1, 0.1)
    plt.yticks(y_ticks)
    ax = plt.gca()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(y_ticks)],
                       fontname='Times New Roman', fontweight='bold')
    # 手动设置横坐标刻度
    x_ticks = np.arange(2, 8, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontname='Times New Roman', fontweight='bold')

    # 设置刻度线加粗
    ax.tick_params(width=2)
    # 设置边框线变粗
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.show()
    fig.savefig('figure_1.png', dpi=600)
    return model, coef, intercept


# 定义误差函数，最小化每对图片荧光数值差异的方差
def error_function(params):
    k, d = params
    calculated_fluorescence_values = np.array([process_image(image_path, k, d) for image_path in image_files])
    # 使用线性回归拟合
    model = LinearRegression()
    x = np.log10([400, 400, 400, 400, 400, 400, 400, 400,
                  3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                  30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000,
                  300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
                  3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000]).reshape(-1, 1)
    y = calculated_fluorescence_values
    model.fit(x, y)
    predicted_values = model.predict(x)
    # 返回均方误差
    return np.mean((predicted_values - y) ** 2)


if __name__ == '__main__':
    base_path = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\QD_Water'
    folders = ['400', '3000', '30000', '300000', '3000000']
    image_files = []

    # 构建图像文件路径列表
    for folder in folders:
        for i in range(1, 9):
            image_files.append(os.path.join(base_path, folder, f'image_{i}.bmp'))

    # 优化阈值 k 和 d
    result = minimize(error_function, x0=np.array([1.17, 20.2]), bounds=[(0, 255), (0, 20)])
    optimal_k, optimal_d = result.x

    # 打印最佳阈值 k 和 d
    print(f"Optimal threshold k: {optimal_k}")
    print(f"Optimal divisor d: {optimal_d}")

    # 计算最佳阈值下每张图片的荧光数值
    calculated_fluorescence_values = [process_image(image_path, optimal_k, optimal_d) for image_path in image_files]

    # 打印每张图片的荧光数值
    for i, value in enumerate(calculated_fluorescence_values):
        print(f"Image {i + 1}: {value}")

    main_directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\QD_Water'
    log_labels, mean_values, std_values, all_image_names, all_fluorescence_values, all_log_labels = get_fluorescence_statistics(
        main_directory, optimal_k, optimal_d)

    # 选择400到3000000文件夹的数据进行线性回归拟合
    filtered_indices = [i for i, label in enumerate(log_labels) if 400 <= 10 ** label <= 30000000]
    filtered_log_labels = [log_labels[i] for i in filtered_indices]
    filtered_mean_values = [mean_values[i] for i in filtered_indices]

    model, coef, intercept = plot_fluorescence_statistics(log_labels, mean_values, std_values, filtered_log_labels,
                                                          filtered_mean_values)

    # 计算预测的 log_label
    all_fluorescence_values_scaled = (np.array(all_fluorescence_values) * 10) - 10
    predicted_log_labels = (all_fluorescence_values_scaled - intercept) / coef
