import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置 matplotlib 支持中文，使用宋体字体
plt.rcParams['font.family'] = ['SimSun']
plt.rcParams['font.sans-serif'] = ['SimSun'] + plt.rcParams['font.sans-serif']
# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 从test2.csv文件中读取数据
df = pd.read_csv('acc.csv')

# 提取海深信息作为横坐标
df["海深"] = df["备注"].str.extract(r'd(\d+)').astype(int)

# 提取测试平均准确度-距离和测试平均准确率-深度中的数值
df["测试平均准确度-距离"] = df["测试平均准确度-距离"].str.extract(r'(\d+\.?\d*)').astype(float)
df["测试平均准确度-深度"] = df["测试平均准确度-深度"].str.extract(r'(\d+\.?\d*)').astype(float)

# 定义颜色、标记和线型
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's']  # 不同的标记用于区分距离和深度
linestyles = ['-', '--', '-.', ':']  # 不同的线型

# 绘图
models = ["mtl_cnn", "mtl_unet", "mtl_unet_cbam", "xception"]
# 设置图片大小以适应 1920x1080 屏幕
plt.figure(figsize=(19.2, 10.8))
for i, model in enumerate(models):
    model_df = df[df["模型名称"] == model]
    color = colors[i]
    linestyle = linestyles[i]
    # 绘制距离和深度准确度
    plt.plot(model_df["海深"], model_df["测试平均准确度-距离"], label=f"{model} - 距离准确度", color=color, marker=markers[0], linestyle=linestyle)
    plt.plot(model_df["海深"], model_df["测试平均准确度-深度"], label=f"{model} - 深度准确度", color=color, marker=markers[1], linestyle=linestyle)

    # 计算综合正确率
    combined_accuracy = (model_df["测试平均准确度-距离"].mean() + model_df["测试平均准确度-深度"].mean()) / 2
    # 找到海深的最大和最小值
    min_depth = model_df["海深"].min()
    max_depth = model_df["海深"].max()
    # 绘制综合正确率点
    # 调整标记大小，这里设置为 200，你可以根据需要修改
    plt.scatter([max_depth + 100], [combined_accuracy], color=color, marker='*', s=200, label=f"{model} - 综合正确率")

# 设置横纵坐标字体大小
plt.rcParams.update({'font.size': 16})  # 增大字体大小
plt.xlabel("海深 (m)", fontsize=16)  # 增大 xlabel 字体大小
plt.xticks(fontsize=16)  # 增大 x 轴刻度字体大小
plt.ylabel("正确率 (%)", fontsize=16)  # 增大 ylabel 字体大小
plt.yticks(fontsize=16)  # 增大 y 轴刻度字体大小
plt.title("不同模型在不同海深下的测试准确度", fontsize=16)
plt.legend()
plt.grid(True)
# 保存为高清图片，分辨率 600 dpi
plt.savefig('正确率.png', dpi=600)  
plt.show()

# 创建雷达图
plt.figure(figsize=(19.2, 10.8))
# 获取唯一的海深值
unique_depths = df["海深"].unique()
# 增加综合正确率维度，所以要多一个角度
angles = np.linspace(0, 2 * np.pi, len(unique_depths) * 2 + 1, endpoint=False).tolist()
angles += angles[:1]

# 为每个海深添加距离和深度的标签，再加上综合正确率标签
labels = []
for depth in unique_depths:
    labels.extend([f"{depth}-距离", f"{depth}-深度"])
labels.append("综合正确率")

# 准备数据
for i, model in enumerate(models):
    model_df = df[df["模型名称"] == model]
    accuracies = []
    for depth in unique_depths:
        depth_df = model_df[model_df["海深"] == depth]
        accuracies.extend([depth_df["测试平均准确度-距离"].mean(), depth_df["测试平均准确度-深度"].mean()])
    # 计算综合正确率
    combined_accuracy = (model_df["测试平均准确度-距离"].mean() + model_df["测试平均准确度-深度"].mean()) / 2
    accuracies.append(combined_accuracy)
    accuracies += accuracies[:1]

    # 绘制雷达图
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, accuracies, 'o-', label=f'{model}', color=colors[i])
    ax.fill(angles, accuracies, alpha=0.2, color=colors[i])
    # 突出显示综合正确率，将最后一个点（综合正确率）的标记改为 *
    ax.plot(angles[-2], accuracies[-2], '*', markersize=20, color=colors[i])

# 设置标签和标题
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=16)  # 增大雷达图 x 轴标签字体大小
ax.tick_params(axis='y', labelsize=16)  # 增大雷达图 y 轴刻度字体大小
ax.set_title('各模型在不同海深下的能力雷达图', fontsize=16)

# 将图例放到更左下方
ax.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
# 保存为高清图片，分辨率 600 dpi
plt.savefig('能力图.png', dpi=600)  
plt.show()