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

import os


def rename_mtb_files(folder_path):
    # 遍历文件夹A下的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            # 在每个子文件夹中查找mtb文件并进行重命名
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.mtb'):
                    file_path = os.path.join(dir_path, file_name)
                    new_file_name = file_name[:-7] + '000.mtb'  # 修改文件名（除去扩展名）的最后三位
                    new_file_path = os.path.join(folder_path, new_file_name)

                    # 检查新文件名是否已存在，避免覆盖已有文件
                    i = 1
                    while os.path.exists(new_file_path):
                        new_file_name = file_name[:-7] + f'{i:03}.mtb'
                        new_file_path = os.path.join(folder_path, new_file_name)
                        i += 1

                    # 重命名mtb文件
                    os.rename(file_path, new_file_path)


# 指定文件夹A的路径
folder_A_path = r'D:\OpenSim\Datasetall\zzhangjiangdata\fall_detection\MTB\heleshen'

# 调用函数进行重命名操作
rename_mtb_files(folder_A_path)




