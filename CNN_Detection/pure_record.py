# -*- coding: utf-8 -*-
# @Time    : 2024/7/9 18:28
# @Author  : Jay
# @File    : pure_record.py
# @Project: QD_detection
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
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


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []
    image_names = []
    data_types = []  # 训练数据或测试数据标识
    with torch.no_grad():
        for images, labels, names in dataloader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
            image_names.extend(names)
    return np.array(predictions).flatten(), np.array(actuals).flatten(), image_names


if __name__ == '__main__':
    device = torch.device('cpu')

    data_dir = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data'
    category = 'FITC'
    pretrained_path = 'best_model_resnet18_FITC.pth'
    image_size = (2048, 2048)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

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
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    # 评估模型
    train_predictions, train_actuals, train_names = evaluate_model(model, train_loader, device)
    test_predictions, test_actuals, test_names = evaluate_model(model, test_loader, device)

    # 保存数据到 Excel 文件
    train_df = pd.DataFrame({
        'Category': [category] * len(train_names),
        'Data Type': ['Train'] * len(train_names),
        'Image Name': train_names,
        'Actual': 10 ** train_actuals,
        'Predicted': 10 ** train_predictions,
        'Log Actual': train_actuals,
        'Log Predicted': train_predictions
    })

    test_df = pd.DataFrame({
        'Category': [category] * len(test_names),
        'Data Type': ['Test'] * len(test_names),
        'Image Name': test_names,
        'Actual': 10 ** test_actuals,
        'Predicted': 10 ** test_predictions,
        'Log Actual': test_actuals,
        'Log Predicted': test_predictions
    })

    result_df = pd.concat([train_df, test_df])

    result_df.to_excel('FITC_predictions_results.xlsx', index=False)

    train_mse = np.mean((train_predictions - train_actuals) ** 2)
    test_mse = np.mean((test_predictions - test_actuals) ** 2)

    print(f'Train MSE: {train_mse:.4f}')
    print(f'Test MSE: {test_mse:.4f}')
    print('Predictions saved successfully.')

