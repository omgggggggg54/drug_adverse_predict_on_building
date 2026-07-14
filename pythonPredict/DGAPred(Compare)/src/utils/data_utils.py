# 导入所需的库
import pandas as pd
import numpy as np
from utils.clac_dis_mesh_sim import cal_SimilarityByMeSHDAG
from alive_progress import alive_bar


def data_feature(data: pd.DataFrame, entity_ids, entity_column: str) -> pd.DataFrame:
    """筛选指定实体，并按实体 ID 排序以保持缓存顺序稳定。"""
    return data[data[entity_column].isin(entity_ids)].sort_values(entity_column).reset_index(drop=True)

def jaccard_similarity(X):
    """计算输入矩阵自身的 Jaccard 相似度矩阵。"""
    import time
    from sklearn.metrics import pairwise_distances

    X_arr = X.values if hasattr(X, 'values') else X
    X_bool = np.asarray(X_arr, dtype=bool)
    n = X_arr.shape[0]
    start = time.time()
    print(f"[Jaccard] 开始计算自相似度矩阵，规模: {n}x{n}，特征维度: {X_arr.shape[1]}")

    block = max(32, min(256, n // 8 if n >= 128 else n))
    Sim = np.zeros((n, n), dtype=float)

    with alive_bar(n, title='[Jaccard] 计算进度', spinner='dots_waves2') as bar:
        for i in range(0, n, block):
            j_end = min(i + block, n)
            distances = pairwise_distances(X_bool[i:j_end], X_bool, metric='jaccard', n_jobs=1)
            Sim[i:j_end, :] = 1.0 - distances
            bar(j_end - i)

    np.fill_diagonal(Sim, 1.0)
    Sim[np.isnan(Sim)] = 0
    print(f"[Jaccard] 完成，总耗时: {time.time() - start:.1f}s")
    return Sim

def Convert_triplelist2matrix(data: pd.DataFrame, pivot_cols):
    """将三列实体关系转换为以 0 填充的矩阵。"""
    matrix = data.pivot(index=pivot_cols[0], columns=pivot_cols[1], values=pivot_cols[2])
    return matrix.sort_index(axis=0).sort_index(axis=1).fillna(0)


def _tanimoto_matrix(fingerprints, title):
    """根据指纹列表计算 Tanimoto 相似度矩阵。"""
    from rdkit import DataStructs

    n = len(fingerprints)
    sim_mat = np.zeros([n, n], dtype=float)
    with alive_bar(n, title=title) as bar:
        for i in range(n):
            bar()
            sim_mat[i][i] = 1.0
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                sim_mat[i][j] = sim
                sim_mat[j][i] = sim
    return sim_mat


def get_RDKit_Similarity(data: pd.DataFrame):
    """基于 RDKit 指纹计算药物相似度矩阵。"""
    from rdkit import Chem

    print("开始计算RDKit指纹相似性，总数：", str(data.shape[0]))
    fingerprints = []
    with alive_bar(data.shape[0], title="生成RDKit指纹") as bar:
        for i in range(data.shape[0]):
            bar()
            mol = Chem.MolFromSmiles(data.loc[i]["SMILES"])
            fingerprints.append(Chem.RDKFingerprint(mol))
    return _tanimoto_matrix(fingerprints, "RDKit相似度")


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
