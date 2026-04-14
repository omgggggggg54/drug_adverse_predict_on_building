
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit import RDLogger

# 抑制RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

# 最终相似度阈值：只保留相似度高于此值的匹配结果
# 建议值：0.75-0.85，根据数据质量调整
FINAL_SIMILARITY_THRESHOLD = 1

old_dir = os.getcwd()
# 获取脚本所在目录的父目录（项目根目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.dirname(script_dir)
print(f'change from {old_dir} to {work_dir}')
os.chdir(work_dir)

# 1. 读取带有SMILES的数据
print("正在读取带有SMILES的数据...")
go_filtered = pd.read_csv(r'data\processed\CTD_chem_go_enriched_withsmiles_pubchem.csv')
pathway_filtered = pd.read_csv(r'data\processed\CTD_chem_pathways_enriched_withsmiles_pubchem.csv')

print(f"GO数据量: {len(go_filtered)}")
print(f"Pathway数据量: {len(pathway_filtered)}")

# 2. 读取 pert_id 与 SMILES 的映射关系
pert_smiles_map = pd.read_csv(r'data\drug_pert_similes_list.csv', sep='\t')
print(f"LINCS数据量: {len(pert_smiles_map)}")

def smiles_to_maccs(smiles):
    """将SMILES转换为MACCS指纹"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return MACCSkeys.GenMACCSKeys(mol)
        return None
    except:
        return None

def calculate_tanimoto_similarity(fp1, fp2):
    """计算Tanimoto相似度"""
    if fp1 is not None and fp2 is not None:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    return 0.0

def find_best_match_batch(target_fp, reference_fps_dict, similarity_threshold=0.8):
    """使用预计算的指纹找到最佳匹配的pert_id"""
    if target_fp is None:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    for pert_id, ref_fp in reference_fps_dict.items():
        if ref_fp is not None:
            similarity = calculate_tanimoto_similarity(target_fp, ref_fp)
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = pert_id
    
    return best_match, best_similarity

# 3. 首先尝试直接匹配相同的SMILES
print("\n步骤1: 直接匹配相同SMILES...")
go_direct_match = pd.merge(go_filtered, pert_smiles_map, on='SMILES', how='inner')
pathway_direct_match = pd.merge(pathway_filtered, pert_smiles_map, on='SMILES', how='inner')

print(f"GO直接匹配数量: {len(go_direct_match)}")
print(f"Pathway直接匹配数量: {len(pathway_direct_match)}")

# 4. 对未匹配的数据使用MACCS指纹相似度匹配
print("\n步骤2: 使用MACCS指纹相似度匹配未匹配的数据...")

# 找出未直接匹配的数据
go_unmatched = go_filtered[~go_filtered['SMILES'].isin(go_direct_match['SMILES'])]
pathway_unmatched = pathway_filtered[~pathway_filtered['SMILES'].isin(pathway_direct_match['SMILES'])]

print(f"GO未匹配数量: {len(go_unmatched)}")
print(f"Pathway未匹配数量: {len(pathway_unmatched)}")

# 预先计算pert_smiles_map的所有MACCS指纹（避免重复计算）
print("预先计算参考数据的MACCS指纹...")
pert_fps_dict = {}
for idx, row in pert_smiles_map.iterrows():
    pert_id = row['pert_id']
    smiles = row['SMILES']
    fp = smiles_to_maccs(smiles)
    pert_fps_dict[pert_id] = fp
print(f"已计算 {len(pert_fps_dict)} 个参考指纹")

# 对未匹配的数据提取唯一SMILES进行匹配，然后广播结果
def match_unique_smiles_and_broadcast(unmatched_df, df_name):
    """对唯一的SMILES进行匹配，然后广播到所有原始行"""
    # 提取唯一的SMILES
    unique_smiles = unmatched_df['SMILES'].unique()
    print(f"{df_name}唯一SMILES数量: {len(unique_smiles)} (原始数量: {len(unmatched_df)})")
    
    # 对唯一SMILES进行匹配
    smiles_match_dict = {}  # SMILES -> (pert_id, similarity)
    processed_count = 0
    
    print(f"处理{df_name}唯一SMILES相似度匹配...")
    for smiles in unique_smiles:
        target_fp = smiles_to_maccs(smiles)
        best_pert_id, similarity = find_best_match_batch(target_fp, pert_fps_dict)
        if best_pert_id is not None:
            smiles_match_dict[smiles] = (best_pert_id, similarity)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"已处理{df_name}唯一SMILES {processed_count}/{len(unique_smiles)}")
    
    print(f"{df_name}匹配到 {len(smiles_match_dict)} 个唯一SMILES")
    
    # 将匹配结果广播回所有原始行
    if smiles_match_dict:
        # 创建匹配结果DataFrame
        match_results = []
        for smiles, (pert_id, similarity) in smiles_match_dict.items():
            match_results.append({'SMILES': smiles, 'pert_id': pert_id, 'similarity': similarity})
        match_df = pd.DataFrame(match_results)
        
        # 通过SMILES合并，将匹配结果广播到所有原始行
        matched_df = unmatched_df.merge(match_df, on='SMILES', how='inner')
        return matched_df
    else:
        return pd.DataFrame()

# 处理GO数据的相似度匹配
go_similarity_matched = match_unique_smiles_and_broadcast(go_unmatched, "GO")

# 处理Pathway数据的相似度匹配
pathway_similarity_matched = match_unique_smiles_and_broadcast(pathway_unmatched, "Pathway")

# 5. 合并直接匹配和相似度匹配的结果
print("\n步骤3: 合并匹配结果...")

# 为直接匹配的数据添加similarity列
go_direct_match['similarity'] = 1.0
pathway_direct_match['similarity'] = 1.0

# 合并结果
if not go_similarity_matched.empty:
    go_final = pd.concat([go_direct_match, go_similarity_matched], ignore_index=True)
else:
    go_final = go_direct_match

if not pathway_similarity_matched.empty:
    pathway_final = pd.concat([pathway_direct_match, pathway_similarity_matched], ignore_index=True)
else:
    pathway_final = pathway_direct_match

print(f"GO最终匹配数量: {len(go_final)}")
print(f"Pathway最终匹配数量: {len(pathway_final)}")

# 6. 只保留需要的列并去重
go_result = go_final[['pert_id', 'GOTermID','similarity']].drop_duplicates()
pathway_result = pathway_final[['pert_id', 'PathwayID','similarity']].drop_duplicates()

print(f"GO去重后数量: {len(go_result)}")
print(f"Pathway去重后数量: {len(pathway_result)}")

# 7. 根据相似度阈值筛选结果
print(f"\n步骤4: 根据相似度阈值筛选 (阈值: {FINAL_SIMILARITY_THRESHOLD})...")
go_before_filter = len(go_result)
pathway_before_filter = len(pathway_result)

go_result = go_result[go_result['similarity'] >= FINAL_SIMILARITY_THRESHOLD]
pathway_result = pathway_result[pathway_result['similarity'] >= FINAL_SIMILARITY_THRESHOLD]

go_filtered_count = go_before_filter - len(go_result)
pathway_filtered_count = pathway_before_filter - len(pathway_result)

print(f"GO筛选后数量: {len(go_result)} (过滤掉 {go_filtered_count} 条低相似度记录)")
print(f"Pathway筛选后数量: {len(pathway_result)} (过滤掉 {pathway_filtered_count} 条低相似度记录)")

# 打印相似度统计
if not go_result.empty:
    print(f"\nGO相似度统计: 最小值={go_result['similarity'].min():.4f}, 最大值={go_result['similarity'].max():.4f}, 平均值={go_result['similarity'].mean():.4f}")
if not pathway_result.empty:
    print(f"Pathway相似度统计: 最小值={pathway_result['similarity'].min():.4f}, 最大值={pathway_result['similarity'].max():.4f}, 平均值={pathway_result['similarity'].mean():.4f}")

go_result.drop(columns=['similarity'], inplace=True)
pathway_result.drop(columns=['similarity'], inplace=True)

# 8. 保存结果
output_dir = r"data\processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

go_output_path = os.path.join(output_dir, "CTD_chem_go_filtered_with_pertid.csv")
pathway_output_path = os.path.join(output_dir, "CTD_chem_pathways_filtered_with_pertid.csv")

go_result.to_csv(go_output_path, index=False)
pathway_result.to_csv(pathway_output_path, index=False)

print("\n处理完成！已保存文件：")
print(f"1. {go_output_path}")
print(f"2. {pathway_output_path}")

# 打印预览
print("\nGO 数据预览:")
print(go_result.head())
print("\nPathway 数据预览:")
print(pathway_result.head())

# 打印匹配统计
print(f"\n匹配统计:")
print(f"GO数据: 直接匹配 {len(go_direct_match)}, 相似度匹配 {len(go_similarity_matched)}, 总计 {len(go_final)}")
print(f"Pathway数据: 直接匹配 {len(pathway_direct_match)}, 相似度匹配 {len(pathway_similarity_matched)}, 总计 {len(pathway_final)}")



