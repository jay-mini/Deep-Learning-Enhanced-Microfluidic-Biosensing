# -*- coding: utf-8 -*-
# @Time    : 2024/7/5 22:25
# @Author  : Jay
# @File    : visualization.py
# @Project: main.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, image_names, transform=None):
        self.images = images
        self.labels = labels
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image_name = self.image_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, image_name


# 预训练的ResNet18模型并修改最后一层
class ResNetRegressor(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        if pretrained_path:
            self.resnet.load_state_dict(torch.load(pretrained_path))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        return self.resnet(x)


def load_data(data_dir, category, image_size):
    images = []
    labels = []
    image_names = []
    category_path = os.path.join(data_dir, category)
    for folder in os.listdir(category_path):
        folder_path = os.path.join(category_path, folder)
        if os.path.isdir(folder_path):
            try:
                label = np.log10(int(folder))
            except ValueError:
                continue
            for filename in os.listdir(folder_path):
                if filename.endswith('.bmp'):
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path).convert('RGB').resize(image_size)
                    images.append(image)
                    labels.append(label)
                    image_names.append(filename)
    return images, np.array(labels), image_names


def evaluate_model(model, dataloader, device, category):
    model.eval()
    predictions = []
    actuals = []
    image_names = []
    categories = []
    with torch.no_grad():
        for images, labels, names in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            image_names.extend(names)
            categories.extend([category] * len(names))
    return np.array(predictions).flatten(), np.array(actuals).flatten(), image_names, categories


if __name__ == '__main__':
    device = torch.device('cpu')

    data_dir = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data'
    categories = ['PBS-Buffer', 'Milk', 'Chicken']
    pretrained_paths = {
        'PBS-Buffer': 'best_model_resnet18_all.pth',
        'Milk': 'best_model_resnet18_Milk.pth',
        'Chicken': 'best_model_resnet18_Chicken.pth'
    }
    image_size = (2048, 2048)
    colors = {'PBS-Buffer': 'blue', 'Milk': 'green', 'Chicken': 'red'}
    markers = {'PBS-Buffer': 'o', 'Milk': 'P', 'Chicken': 's'}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_train_predictions = []
    all_train_actuals = []
    all_test_predictions = []
    all_test_actuals = []
    all_train_labels = []
    all_test_labels = []

    # 初始化字典来保存每个类别的真实值和预测值
    results = {
        category: {
            'train_actuals': [],
            'train_predictions': [],
            'train_names': [],
            'train_category': [],
            'test_actuals': [],
            'test_predictions': [],
            'test_names': [],
            'test_category': []
        }
        for category in categories
    }

    for category in categories:
        images, labels, image_names = load_data(data_dir, category, image_size)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test, train_names, test_names = train_test_split(images, labels, image_names,
                                                                                     test_size=0.2, random_state=43)
        # 创建数据集和数据加载器
        train_dataset = CustomDataset(X_train, y_train, train_names, transform=transform)
        test_dataset = CustomDataset(X_test, y_test, test_names, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 实例化模型并加载训练好的参数
        model = ResNetRegressor(pretrained_path='resnet18_pretrained.pth')
        model.load_state_dict(torch.load(pretrained_paths[category]))
        model.to(device)

        train_predictions, train_actuals, train_names, train_categories = evaluate_model(model, train_loader, device,
                                                                                         category)
        test_predictions, test_actuals, test_names, test_categories = evaluate_model(model, test_loader, device,
                                                                                     category)

        all_train_predictions.extend(train_predictions)
        all_train_actuals.extend(train_actuals)
        all_test_predictions.extend(test_predictions)
        all_test_actuals.extend(test_actuals)
        all_train_labels.extend([category] * len(train_actuals))
        all_test_labels.extend([category] * len(test_actuals))

        # 保存每个类别的真实值和预测值
        results[category]['train_actuals'].extend(10 ** train_actuals)
        results[category]['train_predictions'].extend(10 ** train_predictions)
        results[category]['train_names'].extend(train_names)
        results[category]['train_category'].extend(train_categories)
        results[category]['test_actuals'].extend(10 ** test_actuals)
        results[category]['test_predictions'].extend(10 ** test_predictions)
        results[category]['test_names'].extend(test_names)
        results[category]['test_category'].extend(test_categories)

    all_train_predictions = np.array(all_train_predictions)
    all_train_actuals = np.array(all_train_actuals)
    all_test_predictions = np.array(all_test_predictions)
    all_test_actuals = np.array(all_test_actuals)

    train_mse = np.mean((all_train_predictions - all_train_actuals) ** 2)
    test_mse = np.mean((all_test_predictions - all_test_actuals) ** 2)

    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE: {test_mse:.4f}')

    # 保存数据到 Excel 文件
    with pd.ExcelWriter('predictions_results.xlsx') as writer:
        for category in categories:
            df_train = pd.DataFrame({
                'Image Name': results[category]['train_names'],
                'Category': results[category]['train_category'],
                'Actual': results[category]['train_actuals'],
                'Predicted': results[category]['train_predictions']
            })
            df_test = pd.DataFrame({
                'Image Name': results[category]['test_names'],
                'Category': results[category]['test_category'],
                'Actual': results[category]['test_actuals'],
                'Predicted': results[category]['test_predictions']
            })
            df_train.to_excel(writer, sheet_name=f'{category}_Train', index=False)
            df_test.to_excel(writer, sheet_name=f'{category}_Test', index=False)

    import matplotlib.pyplot as plt
    import numpy as np

    # 假设 categories, colors, markers, all_train_labels, all_train_actuals, all_train_predictions, all_test_labels, all_test_actuals, all_test_predictions 已经定义

    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 22  # 设置全局图例字体大小

    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), sharey=True)

    # 训练集可视化
    for category in categories:
        indices = [i for i in range(len(all_train_labels)) if all_train_labels[i] == category]
        ax1.scatter(all_train_actuals[indices], all_train_predictions[indices], c=colors[category],
                    label=f'{category}', marker=markers[category], s=75)
    ax1.plot([all_train_actuals.min(), all_train_actuals.max()], [all_train_actuals.min(), all_train_actuals.max()],
             'k--', lw=2)
    # ax1.set_xlabel('$\\mathrm{Actual} \\; \\mathrm{Log_{10}} \\; \\mathit{E.coli}$', fontsize=18, fontweight='bold')
    ax1.tick_params('x', labelsize=15)
    ax1.tick_params('y', labelsize=15)
    ax1.legend()  # 这里不需要再设置 fontsize

    # 手动设置纵坐标刻度，每隔一个标注一次
    y_ticks = np.arange(0.5, 8.5, 1)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(y_ticks)],
                        fontname='Times New Roman', fontweight='bold')

    # 手动设置横坐标刻度
    x_ticks = np.arange(0.5, 8.5, 1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks, fontname='Times New Roman', fontweight='bold')

    # 设置刻度线加粗
    ax1.tick_params(width=2)
    # 设置边框线变粗
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)

    # 测试集可视化
    for category in categories:
        indices = [i for i in range(len(all_test_labels)) if all_test_labels[i] == category]
        ax2.scatter(all_test_actuals[indices], all_test_predictions[indices], c=colors[category],
                    label=f'{category}', marker=markers[category], s=75)
    ax2.plot([all_test_actuals.min(), all_test_actuals.max()], [all_test_actuals.min(), all_test_actuals.max()],
             'k--', lw=2)
    ax2.set_xlabel('$\\mathrm{Actual} \\; \\mathrm{Log_{10}} \\; \\mathit{E.coli}$', fontsize=18, fontweight='bold')
    ax2.tick_params('x', labelsize=15)
    ax2.tick_params('y', labelsize=15)
    ax2.legend()  # 这里不需要再设置 fontsize

    # 手动设置纵坐标刻度，每隔一个标注一次
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([f'{tick:.1f}' if i % 2 == 0 else '' for i, tick in enumerate(y_ticks)],
                        fontname='Times New Roman', fontweight='bold')
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.tick_right()

    # 手动设置横坐标刻度
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_ticks, fontname='Times New Roman', fontweight='bold')

    # 设置刻度线加粗
    ax2.tick_params(width=2)
    # 设置边框线变粗
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)

    # 在两个子图之间设置共享的y轴标签
    fig.text(0.04, 0.5,
             '$\\mathrm{Predicted} \\; \\mathrm{Log_{10}} \\; \\mathit{E.coli} \\; \\mathrm{by} \\; \\mathrm{CNN} \\; \\mathrm{based} \\; \\mathrm{Model}$',
             va='center',
             rotation='vertical', fontsize=18, fontweight='bold', fontname='Times New Roman')

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    plt.show()
    fig.savefig('figure_8.png', dpi=600)
    plt.close()


