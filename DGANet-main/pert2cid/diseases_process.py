import pandas as pd
import os

old_dir = os.getcwd()
# 获取脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(script_dir)
print(f'change from {old_dir} to {work_dir}')
os.chdir(work_dir)

# 1. 读取 sider_pert_mesh_list.csv 获取有效的 MESH_ID 集合
print("正在读取 sider_pert_mesh_list.csv...")
sider_mesh = pd.read_csv(r'data\sider_pert_mesh_list.csv', sep='\t')
valid_mesh_ids = set(sider_mesh['MESH_ID'].unique())
print(f"有效的 MESH_ID 数量: {len(valid_mesh_ids)}")

# 2. 读取 CTD_diseases_pathways.csv
print("\n正在读取 CTD_diseases_pathways.csv...")
ctd_diseases = pd.read_csv(r'data\CTD_diseases_pathways.csv')
print(f"原始数据量: {len(ctd_diseases)}")

# 3. 筛选：只保留 DiseaseID 在有效 MESH_ID 集合中的记录
print("\n正在筛选数据...")
ctd_diseases_filtered = ctd_diseases[ctd_diseases['DiseaseID'].isin(valid_mesh_ids)]
print(f"筛选后数据量: {len(ctd_diseases_filtered)}")

# 4. 只保留 DiseaseID 和 PathwayID 列
print("\n提取需要的列...")
result = ctd_diseases_filtered[['DiseaseID', 'PathwayID']].copy()

# 5. 去重
print("去重处理...")
result = result.drop_duplicates()
print(f"去重后数据量: {len(result)}")

# 6. 保存结果
output_path = r'data\processed\CTD_diseases_pathways_filtered.csv'
print(f"\n保存结果到: {output_path}")
result.to_csv(output_path, index=False)

print("\n处理完成！")
print(f"最终结果预览:")
print(result.head(10))
