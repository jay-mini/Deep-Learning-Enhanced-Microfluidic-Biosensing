# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 示例数据
x = np.arange(8).reshape(-1, 1)  # 自变量
y = np.arange(8)  # 因变量

# 线性回归模型拟合
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# 手动计算 R^2
y_mean = np.mean(y)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r_squared_manual = 1 - (ss_res / ss_tot)

# 使用 sklearn 计算 R^2
r_squared_sklearn = model.score(x, y)

print(f"手动计算的 R^2: {r_squared_manual:.3f}")
print(f"sklearn 计算的 R^2: {r_squared_sklearn:.3f}")

# 绘制数据点和拟合线
plt.scatter(x, y, color='blue', label='实际值')
plt.plot(x, y_pred, color='red', label='拟合线')
plt.xlabel('自变量')
plt.ylabel('因变量')
plt.legend()
plt.show()


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
