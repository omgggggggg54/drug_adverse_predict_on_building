# 导入所需的库
import pandas as pd # 用于处理数据的工具
import numpy as np
from utils.clac_dis_mesh_sim import cal_SimilarityByMeSHDAG
from alive_progress import alive_bar # 显示循环的进度条工具
from sklearn.metrics import jaccard_score



'''适合列表/以列表展示的特征'''
def data_feature(data: pd.DataFrame, screen_list: list=None,screen_col: str=None, del_screen_col=True) -> pd.DataFrame:
    
    data = data.copy() # 复制数据，避免后续影响原始数据。 
    if len(screen_list)>0: # 如果提供了 pred_labels 参数，则执行该代码块。
        data=pd.concat([data.loc[data[screen_col]==id] for id in screen_list],ignore_index=True)
        data=data.sort_values(by=screen_col,ascending=[True])
        if del_screen_col:  data = data.drop(columns=[screen_col]) # 去掉用于筛选药物/ADR的列。
    elif screen_col:
        data=data.sort_values(by=screen_col,ascending=[True])
        if del_screen_col:  data = data.drop(columns=[screen_col]) # 去掉用于筛选药物/ADR的列。
    return data # 返回最后处理的数据。

def jaccard_similarity(X, Y=None):
    """
    计算Jaccard相似度。
    - 当 Y 为 None 时，返回 X 与自身的相似度矩阵 (n, n)，并带有分块进度与时间日志。
    - 当 Y 不为 None 时，回退到 sklearn 的逐向量 jaccard_score（较少用）。
    """
    import time
    from sklearn.metrics import pairwise_distances

    if Y is None:
        X_arr = X.values if hasattr(X, 'values') else X
        n = X_arr.shape[0]
        start = time.time()
        print(f"[Jaccard] 开始计算自相似度矩阵，规模: {n}x{n}，特征维度: {X_arr.shape[1]}")

        # 分块计算以便打印进度；块大小可按 n 调整
        block = max(32, min(256, n // 8 if n >= 128 else n))
        Sim = np.zeros((n, n), dtype=float)

        # 使用 alive_bar 展示进度（一次推进一个区块的行数）
        with alive_bar(n, title='[Jaccard] 计算进度', spinner='dots_waves2') as bar:
            for i in range(0, n, block):
                j_end = min(i + block, n)
                # 计算距离(0~1)，再转相似度(1-距离)
                D_blk = pairwise_distances(X_arr[i:j_end], X_arr, metric='jaccard', n_jobs=1)
                Sim[i:j_end, :] = 1.0 - D_blk
                # 推进进度：本区块处理的行数
                bar(j_end - i)

        # 对角置1；处理NaN
        np.fill_diagonal(Sim, 1.0)
        Sim[np.isnan(Sim)] = 0
        total_cost = time.time() - start
        print(f"[Jaccard] 完成，总耗时: {total_cost:.1f}s")
        return Sim
    else:
        # Y 不为空：按逐向量 jaccard_score 计算（不常用路径）
        Sim = jaccard_score(X, Y)
        return Sim

def Convert_triplelist2matrix(data: pd.DataFrame,pivot_cols=[],fillna_val=0.0):
    data = data.copy() # 复制数据，避免后续影响原始数据。
    mat_data=data.pivot(index=pivot_cols[0],columns=pivot_cols[1],values=pivot_cols[2])  
    mat_data=mat_data.sort_index(axis=0,ascending=True)
    mat_data=mat_data.sort_index(axis=1,ascending=True)
    mat_data=mat_data.fillna(fillna_val)
    return mat_data

def cal_drug_similarityBySmiles(SMILES1,SMILES2):
    # 惰性导入RDKit，避免Windows多进程环境下的DLL初始化问题
    from rdkit import Chem, DataStructs
    mol1=Chem.MolFromSmiles(SMILES1)
    mol2=Chem.MolFromSmiles(SMILES2)
    # RDKit指纹（此处无需rdMolDescriptors，直接使用RDKFingerprint）
    fps1=Chem.RDKFingerprint(mol1)
    fps2=Chem.RDKFingerprint(mol2)
    return DataStructs.FingerprintSimilarity(fps1,fps2) 

def get_SMILES_Similarity(data: pd.DataFrame):
    print("开始计算SMILES相似性，总数：",str(data.shape[0]))
    smiles_sim_mat = np.zeros([data.shape[0],data.shape[0]], dtype = float, order = 'C')
    with alive_bar(data.shape[0]) as bar:
        for i in  range(data.shape[0]):
            bar()
            Smile_i=data.loc[i]["SMILES"]
            for j in range(i):
                Smile_j = data.loc[j]["SMILES"]
                smiles_sim_mat[i][j]=cal_drug_similarityBySmiles(Smile_i,Smile_j)
    smiles_sim_mat = smiles_sim_mat + smiles_sim_mat.T + np.eye(data.shape[0]) 
    return smiles_sim_mat


def get_SMILES_Similarity_forone(SMILES1,data: pd.DataFrame):
    print("开始计算SMILES相似性，总数：",str(data.shape[0]))
    smiles_sim_mat = np.zeros([1,data.shape[0]], dtype = float, order = 'C')
    with alive_bar(data.shape[0]) as bar:
        for i in  range(data.shape[0]):
            bar()
            Smile_i=data.loc[i]["SMILES"]
            smiles_sim_mat[0][i]=cal_drug_similarityBySmiles(SMILES1,Smile_i)
    return smiles_sim_mat

def get_MESH_Similarity(data: pd.DataFrame):
    print("开始计算MESH相似性，总数：",str(data.shape[0]))
    Mesh_sim_mat = np.zeros([data.shape[0],data.shape[0]], dtype = float, order = 'C')
    with alive_bar(data.shape[0]) as bar:
        for i in  range(data.shape[0]):
            bar()
            Dis_i=eval(data.loc[i]["Dict_MESH"])
            for j in range(i):
                Dis_j = eval(data.loc[j]["Dict_MESH"])
                Mesh_sim_mat[i][j]=cal_SimilarityByMeSHDAG(Dis_i,Dis_j)
    Mesh_sim_mat = Mesh_sim_mat + Mesh_sim_mat.T + np.eye(data.shape[0]) 
    return Mesh_sim_mat

#
