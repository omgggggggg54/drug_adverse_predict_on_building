import pandas as pd
import numpy as np
import os
import requests
import time
import random
from urllib.parse import quote

old_dir = os.getcwd(); 
work_dir = r'C:\Users\yyphoto\Desktop\共享文件\DGANet-main'
print('change from'+old_dir+'to'+work_dir)
os.chdir(work_dir)

# 1. 读取原始数据
print("正在读取数据...")
# 使用 chunksize 防止内存溢出（如果文件非常大），或者直接读取（这里文件似乎还可以接受直接读取）
go = pd.read_csv(r"data\CTD_chem_go_enriched1.csv")
pathway = pd.read_csv(r"data\CTD_chem_pathways_enriched1.csv")

print(f"原始 GO 数据量: {go.shape}")
print(f"原始 Pathway 数据量: {pathway.shape}")

# 2. 数据清洗与筛选
# 设定 P 值阈值，通常选 0.05
p_value_threshold = 0.05

print(f"正在筛选 CorrectedPValue < {p_value_threshold} 的数据...")

# 处理 GO 数据
# 保留 ChemicalID (药物) 和 GOTermID (特征)
go_filtered = go[go['CorrectedPValue'] < p_value_threshold][['ChemicalID', 'ChemicalName', 'GOTermID']].copy()
go_filtered.drop_duplicates(inplace=True) # 去重
print(f"筛选后 GO 数据量: {go_filtered.shape}")

# 处理 Pathway 数据
# 保留 ChemicalID (药物) 和 PathwayID (特征)
pathway_filtered = pathway[pathway['CorrectedPValue'] < p_value_threshold][['ChemicalID', 'ChemicalName', 'PathwayID']].copy()
pathway_filtered.drop_duplicates(inplace=True) # 去重
print(f"筛选后 Pathway 数据量: {pathway_filtered.shape}")

# 取所有化学名
chemical_names_intersection = set(go_filtered['ChemicalName'].tolist()) & set(pathway_filtered['ChemicalName'].tolist())
print(f"GO 和 Pathway 的 ChemicalName 交集: {len(chemical_names_intersection)}")

# 获取化学名列表
names = list(chemical_names_intersection)
print(f"将查询 {len(names)} 个化学名的SMILES")

base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/CSV'
headers = {"content-type":"application/x-www-form-urlencoded"}

# 存储化学名到SMILES的映射
name_to_smiles = {}

def get_smiles_with_retry(name, max_retries=3):
    """获取SMILES，带重试机制"""
    for attempt in range(max_retries):
        try:
            # URL编码处理特殊字符
            encoded_name = quote(name, safe='')
            url = base_url.format(encoded_name)
            
            # 设置超时和重试参数
            res = requests.get(url, headers=headers, timeout=10)
            
            if res.status_code == 200:
                lines = res.text.strip().split('\n')
                if len(lines) > 1 and ',' in lines[1]:
                    smiles = lines[1].split(',')[1].strip('"')
                    return smiles
                return None
            elif res.status_code == 404:
                return None  # 化合物不存在，不重试
            else:
                print(f"HTTP {res.status_code} for {name}, 重试 {attempt + 1}/{max_retries}")
                
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"请求异常 {name} (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 5))  # 异常时等待更长时间
    
    return None

# 主循环
for i, name in enumerate(names):
    # 跳过明显无效的化学名
    if len(name) > 100 or any(x in name.lower() for x in ['agents', 'antimetabolites', 'tetrahydrate']):
        print(f"跳过无效化学名: {name}")
        continue
    
    smiles = get_smiles_with_retry(name)
    if smiles:
        name_to_smiles[name] = smiles
        print(f"✓ {name}: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
    else:
        print(f"✗ 未找到: {name}")
    
    # 自然访问频率控制：随机等待0.8-2.5秒
    wait_time = random.uniform(0.8, 2.5)
    time.sleep(wait_time)
    
    # 每50个请求后稍作休息
    if (i + 1) % 50 == 0:
        print(f"已处理 {i + 1}/{len(names)} 个化学名，休息5秒...")
        time.sleep(5)
    elif (i + 1) % 10 == 0:
        print(f"已处理 {i + 1}/{len(names)} 个化学名")

print(f"\n成功获取 {len(name_to_smiles)} 个化学名的SMILES")

# 将SMILES添加到go_filtered和pathway_filtered中
go_filtered['SMILES'] = go_filtered['ChemicalName'].map(name_to_smiles)
pathway_filtered['SMILES'] = pathway_filtered['ChemicalName'].map(name_to_smiles)

# 只保留有SMILES的记录
go_filtered = go_filtered.dropna(subset=['SMILES'])
pathway_filtered = pathway_filtered.dropna(subset=['SMILES'])

print(f"添加SMILES后，GO数据量: {len(go_filtered)}")
print(f"添加SMILES后，Pathway数据量: {len(pathway_filtered)}")


go_filtered.to_csv(r'data\processed\CTD_chem_go_enriched_withsmiles_pubchem.csv', index=False)
pathway_filtered.to_csv(r'data\processed\CTD_chem_pathways_enriched_withsmiles_pubchem.csv', index=False)