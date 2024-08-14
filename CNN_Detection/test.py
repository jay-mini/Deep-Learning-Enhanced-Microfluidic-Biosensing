# -*- coding: utf-8 -*-
# @Time    : 2024/6/24 12:39
# @Author  : Jay
# @File    : test.py
# @Project: main.py
# 尝试将训练好的网络用于完全没有训练的58, 28, 20, 15, 10的数据
import os.path
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from model_train_qd import CustomDataset, ResNetRegressor
from sklearn.model_selection import train_test_split
from PIL import Image

if __name__ == '__main__':
    device = 'cpu'
    data_dir = r'\Escherichia coli detection\QD_detection\Data\QD'
    folders = [10, 15, 20, 28, 58]
    image_size = (2048, 2048)

    images = []
    labels = []

    for folder in folders:
        folder_path = os.path.join(data_dir, str(folder))
        label = np.log10(folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.bmp'):
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path).convert('RGB').resize(image_size)
                images.append(image)
                labels.append(label)

    # 转换为numpy数组
    labels = np.array(labels)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 创建数据集和数据加载器
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    test_dataset = CustomDataset(X_test, y_test, transform=transform)
    # 打印训练集和测试集的数据量
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 实例化模型
    pretrained_path = 'resnet18_pretrained.pth'
    # model = CNNModel()
    model = ResNetRegressor(pretrained_path=pretrained_path)
    model.to(device)
    print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 保存最佳模型
    best_model_path = r'best_model_resnet18.pth'
    best_loss = float('inf')

    # 加载最佳模型参数
    model.load_state_dict(torch.load(best_model_path))

    # 评估模型
    model.eval()
    test_loss = 0.0
    train_predictions = []
    train_actuals = []
    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            train_predictions.extend(outputs.cpu().numpy())
            train_actuals.extend(labels.cpu().numpy())

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            test_predictions.extend(outputs.cpu().numpy())
            test_actuals.extend(labels.cpu().numpy())

    train_predictions = np.array(train_predictions).flatten()
    train_actuals = np.array(train_actuals).flatten()
    test_predictions = np.array(test_predictions).flatten()
    test_actuals = np.array(test_actuals).flatten()

    print(f'Train Loss: {np.mean((train_predictions - train_actuals) ** 2):.4f}')
    print(f'Test Loss: {np.mean((test_predictions - test_actuals) ** 2):.4f}')

    # 可视化训练集和测试集上的实际值和预测值
    fig = plt.figure(figsize=(7, 6))

    # 训练集可视化
    plt.scatter(train_actuals, train_predictions, c='blue', label='Train Data')
    plt.plot([train_actuals.min(), train_actuals.max()], [train_actuals.min(), train_actuals.max()], 'k--', lw=2)
    plt.scatter(test_actuals, test_predictions, c='red', label='Test Data')
    plt.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], 'k--', lw=2)
    plt.xlabel('Actual log10(C)', fontsize=18)
    plt.ylabel('Predicted log10(C)', fontsize=18)
    plt.tick_params('x', labelsize=18)
    plt.tick_params('y', labelsize=18)
    plt.legend(fontsize=18)

    plt.show()
    fig.savefig('figure_2.png', dpi=600)

