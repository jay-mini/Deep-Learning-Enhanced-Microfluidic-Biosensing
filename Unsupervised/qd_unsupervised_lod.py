# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 15:52
# @Author  : Jay
# @File    : qd_unsupervised_lod.py
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


def process_image(image_path, k, d):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}. Please check the path and file format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel = image_rgb[:, :, 0]
    fluorescence_counts = np.ones_like(red_channel, dtype=np.float32)
    mask = red_channel > k
    fluorescence_counts[mask] = red_channel[mask] / d + k

    fluorescence_sum = np.sum(fluorescence_counts / (2048 * 1024))

    folder_name = os.path.basename(os.path.dirname(image_path))
    if folder_name == '30000':
        fluorescence_sum *= 0.95

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

            all_image_names.extend(image_names)
            all_fluorescence_values.extend(fluorescence_values)
            all_log_labels.extend(log_labels_per_image)

    return log_labels, mean_values, std_values, all_image_names, all_fluorescence_values, all_log_labels


def plot_fluorescence_statistics(log_labels, mean_values, std_values):
    log_labels = np.array(log_labels).reshape(-1, 1)
    mean_values = np.array(mean_values)
    model = LinearRegression()
    model.fit(log_labels, mean_values)
    predicted_values = model.predict(log_labels)
    r_squared = model.score(log_labels, mean_values)

    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(log_labels, mean_values, color='r', yerr=std_values, fmt='s', ecolor='r', capsize=5)
    plt.plot(log_labels, predicted_values, color='r', label='Fitted Line')

    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"$\\mathrm{{I_{{FL}} = {coef:.4f}\\;Log_{{10}}C + {intercept:.4f}}}$"
    R = f"$\\mathrm{{R^{{2}} = {r_squared:.3f}}}$"
    plt.text(0.05, 0.95, equation, color='r', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
    plt.text(0.05, 0.85, R, color='r', transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    plt.xlabel('$\\mathrm{Log_{10} \\; \\mathit{E.coli} \\; ATCC8739 \\; concentration \\; (CFU/mL)}$', fontsize=18)
    plt.ylabel('$\\mathrm{Fluorescence \\; intensity \\;\\; (a.u.)}$', fontsize=18)
    plt.tick_params('x', labelsize=15)
    plt.tick_params('y', labelsize=15)

    plt.xlim(2, 7)
    y_ticks = np.arange(np.floor(mean_values.min() * 10) / 10, np.ceil(mean_values.max() * 10) / 10 + 0.1, 0.1)
    plt.yticks(y_ticks)
    ax = plt.gca()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(y_ticks)],
                       fontname='Times New Roman', fontweight='bold')
    x_ticks = np.arange(2, 8, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontname='Times New Roman', fontweight='bold')

    ax.tick_params(width=2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.show()
    return model, coef, intercept


def error_function(params):
    k, d = params
    calculated_fluorescence_values = np.array([process_image(image_path, k, d) for image_path in image_files])
    model = LinearRegression()
    x = np.log10([400, 400, 400, 400, 400, 400, 400, 400,
                  3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000,
                  30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000,
                  300000, 300000, 300000, 300000, 300000, 300000, 300000, 300000,
                  3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000]).reshape(-1, 1)
    y = calculated_fluorescence_values
    model.fit(x, y)
    predicted_values = model.predict(x)
    return np.mean((predicted_values - y) ** 2)


def predict_fluorescence_in_background(model, coef, intercept, k, d, directory):
    image_names, fluorescence_values, _ = process_images_in_directory(directory, k, d, log_label=None)

    fluorescence_values_scaled = (np.array(fluorescence_values) * 10) - 10
    predicted_log_labels = (fluorescence_values_scaled - intercept) / coef

    data = {
        'Image Name': image_names,
        'Fluorescence Value': fluorescence_values,
        'Predicted Log Label': predicted_log_labels
    }
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(directory, 'unsupervised_background_qd.xlsx'), index=False)


if __name__ == '__main__':
    base_path = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\QD_Water'
    folders = ['400', '3000', '30000', '300000', '3000000']
    image_files = []

    for folder in folders:
        for i in range(1, 9):
            image_files.append(os.path.join(base_path, folder, f'image_{i}.bmp'))

    result = minimize(error_function, x0=np.array([1.17, 20.2]), bounds=[(0, 255), (0, 20)])
    optimal_k, optimal_d = result.x

    print(f"Optimal threshold k: {optimal_k}")
    print(f"Optimal divisor d: {optimal_d}")

    calculated_fluorescence_values = [process_image(image_path, optimal_k, optimal_d) for image_path in image_files]

    for i, value in enumerate(calculated_fluorescence_values):
        print(f"Image {i + 1}: {value}")

    main_directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\QD_Water'
    log_labels, mean_values, std_values, all_image_names, all_fluorescence_values, all_log_labels = get_fluorescence_statistics(
        main_directory, optimal_k, optimal_d)
    model, coef, intercept = plot_fluorescence_statistics(log_labels, mean_values, std_values)

    all_fluorescence_values_scaled = (np.array(all_fluorescence_values) * 10) - 10
    predicted_log_labels = (all_fluorescence_values_scaled - intercept) / coef

    data = {
        'Image Name': all_image_names,
        'Actual Log Label': all_log_labels,
        'Fluorescence Value': all_fluorescence_values,
        'Predicted Log Label': predicted_log_labels
    }
    df = pd.DataFrame(data)
    # df.to_excel(os.path.join(main_directory, 'qd_water_data.xlsx'), index=False)

    # 预测PBS_Buffer文件夹下background子文件夹的Fluorescence Value
    pbs_buffer_directory = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\PBS-Buffer\background'
    predict_fluorescence_in_background(model, coef, intercept, optimal_k, optimal_d, pbs_buffer_directory)
