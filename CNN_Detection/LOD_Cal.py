# -*- coding: utf-8 -*-
# @Time    : 2024/7/14 14:27
# @Author  : Jay
# @File    : LOD_Cal.py
# @Project: QD_detection
# 计算LOD
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, image_names, transform=None):
        self.images = images
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_name = self.image_names[idx]
        if self.transform:
            image = self.transform(image)
        return image, image_name


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
    image_names = []
    category_path = os.path.join(data_dir, category)
    for filename in os.listdir(category_path):
        if filename.endswith('.bmp'):
            image_path = os.path.join(category_path, filename)
            image = Image.open(image_path).convert('RGB').resize(image_size)
            images.append(image)
            image_names.append(filename)
    return images, image_names


def predict_model(model, dataloader, device):
    model.eval()
    predictions = []
    image_names = []
    with torch.no_grad():
        for images, names in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy())
            image_names.extend(names)
    return np.array(predictions).flatten(), image_names


if __name__ == '__main__':
    device = torch.device('cpu')

    data_dir = r'C:\Users\Rubis\Desktop\Documents\Escherichia coli detection\QD_detection\Data\FITC'
    category = 'background'
    pretrained_path = 'best_model_resnet18_FITC.pth'
    image_size = (2048, 2048)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images, image_names = load_data(data_dir, category, image_size)

    # 创建数据集和数据加载器
    dataset = CustomDataset(images, image_names, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 实例化模型并加载训练好的参数
    model = ResNetRegressor(pretrained_path='resnet18_pretrained.pth')
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)

    # 预测模型
    predictions, image_names = predict_model(model, dataloader, device)

    # 保存预测数据到 Excel 文件
    result_df = pd.DataFrame({
        'Category': [category] * len(image_names),
        'Image Name': image_names,
        'Predicted': 10 ** predictions,
        'Log Predicted': predictions
    })

    result_df.to_excel('background_fitc.xlsx', index=False)

    print('Predictions saved successfully.')
