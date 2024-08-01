# -*- coding: utf-8 -*-
# @Time    : 2024/6/23 14:45
# @Author  : Jay
# @File    : model_train_qd.py
# @Project: main.py
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch.nn.functional as F


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.conv1x1_1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)  # 1x1卷积用于残差匹配

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0)  # 1x1卷积用于残差匹配

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.conv1x1_3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)  # 1x1卷积用于残差匹配

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 第一个残差连接
        residual = self.conv1x1_1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        residual = self.pool1(residual)
        x += residual

        # 第二个残差连接
        residual = self.conv1x1_2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        residual = self.pool2(residual)
        x += residual

        # 第三个残差连接
        residual = self.conv1x1_3(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        residual = self.pool3(residual)
        x += residual

        # 展平和全连接层
        x = x.view(-1, 32 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device = torch.device('cpu')
    data_dir = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\FITC'
    folders = [400, 3000, 30000, 300000, 3000000]
    image_size = (2048, 2048)

    # 加载数据
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
    best_model_path = r'best_model_resnet18_FITC.pth'
    best_loss = float('inf')

    # 训练模型
    # num_epochs = 2000
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for images, labels in train_loader:
    #         images = images.to(device)
    #         labels = labels.to(device).float().unsqueeze(1)
    #
    # 前向传播
    # outputs = model(images)
    # loss = criterion(outputs, labels)
    #
    # 反向传播和优化
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    # running_loss += loss.item()
    #
    # epoch_loss = running_loss / len(train_loader)
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # 保存最佳模型
    # if epoch_loss < best_loss:
    #     best_loss = epoch_loss
    #     torch.save(model.state_dict(), best_model_path)

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
    fig = plt.figure(figsize=(12, 5))

    # 训练集可视化
    plt.subplot(1, 2, 1)
    plt.scatter(train_actuals, train_predictions, c='blue', label='Train Data')
    plt.plot([train_actuals.min(), train_actuals.max()], [train_actuals.min(), train_actuals.max()], 'k--', lw=2)
    plt.xlabel('Actual log10(C)')
    plt.ylabel('Predicted log10(C)')
    plt.title('FITC Train Data: Actual vs Predicted')
    plt.legend()

    # 测试集可视化
    plt.subplot(1, 2, 2)
    plt.scatter(test_actuals, test_predictions, c='red', label='Test Data')
    plt.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], 'k--', lw=2)
    plt.xlabel('Actual log10(C)')
    plt.ylabel('Predicted log10(C)')
    plt.title('FITC Test Data: Actual vs Predicted')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # fig.savefig('figure_4.png', dpi=600)
