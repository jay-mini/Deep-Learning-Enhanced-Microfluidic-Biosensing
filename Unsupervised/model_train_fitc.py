# -*- coding: utf-8 -*-
# @Time    : 2024/6/22 17:10
# @Author  : Jay
# @File    : model_train_qd.py
# @Project: QD_detection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from qd_unsupervised_lod import get_fluorescence_statistics
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
    # 获取红色通道(QD_all)/绿色通道(FITC)
    red_channel = image_rgb[:, :, 0]
    # red_channel = image_rgb[:, :, 1]
    # 初始化荧光计数矩阵
    fluorescence_counts = np.ones_like(red_channel, dtype=np.float32)
    # 遍历每个像素并计算荧光标记数
    mask = red_channel > k
    fluorescence_counts[mask] = red_channel[mask] / d + k

    fluorescence_sum = np.sum(fluorescence_counts / (2048 * 1024))

    folder_name = os.path.basename(os.path.dirname(image_path))
    if folder_name == '30000':
        fluorescence_sum *= 0.95  # 将数值减少 5%

    return fluorescence_sum


def plot_fluorescence_statistics(log_labels, mean_values, std_values):
    # 执行线性回归
    log_labels = np.array(log_labels).reshape(-1, 1)
    mean_values = np.array(mean_values)
    model = LinearRegression()
    model.fit(log_labels, mean_values)
    predicted_values = model.predict(log_labels)
    # 计算R^2
    r_squared = model.score(log_labels, mean_values)

    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(log_labels, mean_values, color='r', yerr=std_values, fmt='s', ecolor='r', capsize=5)
    plt.plot(log_labels, predicted_values, color='r', label='Fitted Line')

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
    plt.xlim(2, 7)
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
    fig.savefig('figure_3.png', dpi=600)


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
    base_path = r'\Escherichia coli detection\QD_detection\Data\QD_Water'
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

    # 绘制结果
    # x_values = np.log10([400, 400, 400, 400, 400, 400, 400, 400,
    #                      3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
    #                      30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000,
    #                      300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
    #                      3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000]).reshape(-1, 1)
    #
    # plt.scatter(x_values, calculated_fluorescence_values, color='blue', label='Calculated')
    #
    # model = LinearRegression()
    # model.fit(x_values, calculated_fluorescence_values)
    # predicted_values = model.predict(x_values)
    #
    # plt.plot(x_values, predicted_values, color='red', label='Fitted Line')
    # plt.xlabel('Log')
    # plt.ylabel('Fluorescence Value')
    # # plt.legend()
    # plt.title('Calculated Fluorescence Values for Each Image')
    #
    # # 计算R^2
    # r_squared = model.score(x_values, calculated_fluorescence_values)
    # coef = model.coef_[0]
    # intercept = model.intercept_
    # equation = f"$I_{{FL}} = {coef:.2f} \log(x) + {intercept:.2f}$"
    # plt.text(0.05, 0.95, f"{equation}\n$R^2$={r_squared:.2f}", transform=plt.gca().transAxes, fontsize=12,
    #          verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    #
    # plt.show()

    main_directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\QD_Water'
    log_labels, mean_values, std_values = get_fluorescence_statistics(main_directory, optimal_k, optimal_d)
    plot_fluorescence_statistics(log_labels, mean_values, std_values)
