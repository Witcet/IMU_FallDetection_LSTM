# print("hello github")

# import os
#
# folder_path = r'D:\OpenSim\Datasetall\zzhangjiangdata\fall_detection\MTB\heleshen'  # 文件夹A的路径
#
# for filename in os.listdir(folder_path):
#     if filename.endswith('.mtb'):  # 仅处理以.mtb结尾的文件
#         name_parts = os.path.splitext(filename)  # 分离文件名和扩展名
#         file_number = name_parts[0].split('-')[-1]  # 提取文件名中的数字部分
#         try:
#             new_number = file_number  # 将数字加6，并转换为字符串
#             new_filename = f"{name_parts[0][:-len(file_number)]}0{new_number}{name_parts[1]}"  # 构建新文件名
#             old_path = os.path.join(folder_path, filename)  # 原始文件的完整路径
#             new_path = os.path.join(folder_path, new_filename)  # 新文件的完整路径
#             os.rename(old_path, new_path)  # 重命名文件
#             print(f"重命名文件: {filename} -> {new_filename}")
#         except ValueError:
#             print(f"无效的文件名: {filename}")

import pandas as pd
import matplotlib.pyplot as plt

filename="MT_01210CD7-063-000_00B4D206_101"

# 读取文本文件，跳过文件头
df = pd.read_csv(f'image/{filename}.txt', delim_whitespace=True, skiprows=4)

# 提取需要的字段
fields = ['Acc_X', 'Acc_Y', 'Acc_Z']
data = df[fields]

# 绘制曲线
for field in fields:
    plt.plot(df['PacketCounter'], data[field], label=field)

# 添加图例、标签等
plt.legend()
plt.xlabel('PacketCounter')
plt.title('Sensor Data')

# 保存图像
plt.savefig(f'image/{filename}.png')

# 显示图形
plt.show()








