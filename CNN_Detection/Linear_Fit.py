# -*- coding: utf-8 -*-
# @Time    : 2024/7/12 20:49
# @Author  : Jay
# @File    : Linear_Fit.py
# @Project: QD_detection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 读取Excel文件
file_name = 'test.xlsx'  # 替换为你的文件名
df = pd.read_excel(file_name)

# 假设第一列为 x，第二列为 y
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

# 对数据进行线性拟合
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
r_squared = model.score(x, y)

# 打印结果
print("线性拟合结果: 斜率 = {:.2f}, 截距 = {:.2f}".format(model.coef_[0], model.intercept_))
print("R^2 值:", r_squared)

# 绘图
fig = plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='实际值')
plt.plot(x, y_pred, color='red', label='拟合线')

# 添加方程和R²值
coef = model.coef_[0]
intercept = model.intercept_
equation = f"$\\mathrm{{y = {coef:.4f}x + {intercept:.4f}}}$"
R = f"$\\mathrm{{R^{{2}} = {r_squared:.3f}}}$"
plt.text(0.05, 0.95, equation, color='red', transform=plt.gca().transAxes, fontsize=18, verticalalignment='top')
plt.text(0.05, 0.85, R, color='red', transform=plt.gca().transAxes, fontsize=18, verticalalignment='top')

# 设置图表格式
plt.xlabel('$\\mathrm{Actual} \\; \\mathrm{Log_{10}} \\; \\mathit{E.coli}$', fontsize=18, fontweight='bold')
plt.ylabel('$\\mathrm{Predicted} \\; \\mathrm{Log_{10}} \\; \\mathit{E.coli}$', fontsize=18, fontweight='bold')
# plt.legend()
# plt.grid(True)

# 设置刻度和标签字体
plt.tick_params(axis='both', which='major', labelsize=12)
plt.show()
fig.savefig('figure_10.png', dpi=600)
