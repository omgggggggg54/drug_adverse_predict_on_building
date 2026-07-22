"""共享基因 TF-IDF-SVD 特征缓存。"""

import csv
import json
import os

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from utils.raw_feature_generation import ensure_adr_gene_raw, ensure_drug_gene_raw


CACHE_VERSION = 1
SVD_DIM = 256
MIN_DF = 2
MAX_DF = 0.9
TOP_K = 2048


def _source_state(path):
    """返回用于识别缓存有效性的源文件状态。"""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,#文件大小
        "mtime_ns": stat.st_mtime_ns,#最后修改时间
    }


def _read_vocabulary(path, entity_column, entity_ids):
    """读取当前训练实体对应的去重基因词表。"""
    selected = set(map(str, entity_ids))
    genes = set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if str(row[entity_column]) in selected:
                genes.add(str(row["GeneSymbol"]))
    return sorted(genes)


def _load_token_matrix(path, vocabulary, shared_lookup):
    """把本地基因 token 缓存映射到共享词表空间。"""
    with np.load(path, allow_pickle=False) as cache:
        token_ids = cache["token_ids"].astype(np.int64)
        offsets = cache["offsets"].astype(np.int64)
        ids = cache["ids"].astype(str)

    shared_indices = np.asarray(
        [shared_lookup[gene] for gene in vocabulary], dtype=np.int32
    )[token_ids - 1]#把本地token id映射到共享词表空间
    matrix = sparse.csr_matrix(
        (np.ones(len(shared_indices), dtype=np.float32), shared_indices, offsets),
        shape=(len(ids), len(shared_lookup)),
    )#存成稀疏矩阵,offset是每个实体(行)该为1的列索引范围对应到shared_indices
    return matrix, ids


def _tfidf_matrix(binary_matrix, idf):
    """每个实体的基因关联乘上对应的基因IDF权重，并每个实体只保留权重最高的 TOP_K 个基因,
    返回矩阵形状为 [实体数, 经过MIN_DF,MAX_DF筛选的基因数] 的稀疏矩阵。"""
    weighted = binary_matrix.multiply(idf).tocsr()#每个实体的基因关联（都是 1）乘上对应基因的 IDF 权重
    indices = []
    values = []
    offsets = [0]
    for row in range(weighted.shape[0]):
        start, end = weighted.indptr[row:row + 2]
        columns = weighted.indices[start:end]
        row_values = weighted.data[start:end]
        if len(columns) > TOP_K:#np.argpartition(a, kth)原本位于第k位(从小到大排序)的那个元素，在洗牌后的新数组里，必须百分之百地出现在索引kth位置上
            selected = np.argpartition(row_values, -TOP_K)[-TOP_K:]#取出最大的TOP_K个基因的索引
            columns = columns[selected]#取出对应的基因索引
            row_values = row_values[selected]#取出对应的基因权重
        indices.extend(columns.tolist())
        values.extend(row_values.tolist())
        offsets.append(len(indices))
    weighted = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), np.asarray(indices), np.asarray(offsets)),
        shape=weighted.shape,
    )
    return normalize(weighted, norm="l2", axis=1, copy=False)


def _read_cached_svd(path):
    """读取共享 TF-IDF-SVD 特征缓存。"""
    with np.load(path, allow_pickle=False) as cache:
        return {
            "drug_svd": cache["drug_svd"].astype(np.float32),
            "adr_svd": cache["adr_svd"].astype(np.float32),
        }


def ensure_gene_tfidf_svd(rawpath, similarity_path, drug_ids, adr_ids):
    """构建或读取固定配置的共享基因 TF-IDF-SVD 特征。"""
    rawpath = os.path.normpath(rawpath)
    cache_dir = os.path.join(os.path.normpath(similarity_path), "raw_feature_v2")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "gene_tfidf_svd.npz")
    metadata_path = os.path.join(cache_dir, "gene_tfidf_svd.json")
    drug_source = os.path.join(rawpath, "ctd_chem_pert_gene_ixns_list.csv")
    adr_source = os.path.join(rawpath, "ctd_gene_adr_asso_list_4386.csv")
    cache_key = {
        "version": CACHE_VERSION,
        "drug_source": _source_state(drug_source),
        "adr_source": _source_state(adr_source),
        "drug_ids": list(map(str, drug_ids)),
        "adr_ids": list(map(str, adr_ids)),#转字符串
        "svd_dim": SVD_DIM,
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "top_k": TOP_K,
    }

    if os.path.exists(cache_path) and os.path.exists(metadata_path):#如果缓存有效则读缓存
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("cache_key") == cache_key:
            return _read_cached_svd(cache_path)

    ensure_drug_gene_raw(rawpath, similarity_path, drug_ids, "drug_dgen_raw_tokens.npz")
    ensure_adr_gene_raw(rawpath, similarity_path, adr_ids, "adr_gda_raw_tokens.npz")
    drug_vocabulary = _read_vocabulary(drug_source, "pert_id", drug_ids)
    adr_vocabulary = _read_vocabulary(adr_source, "MESH_ID", adr_ids)
    shared_vocabulary = sorted(set(drug_vocabulary) | set(adr_vocabulary))#取drug-gene和adr-gene的gene并集
    shared_lookup = {gene: index for index, gene in enumerate(shared_vocabulary)}

    drug_matrix, cached_drug_ids = _load_token_matrix(
        os.path.join(similarity_path, "drug_dgen_raw_tokens.npz"),
        drug_vocabulary,
        shared_lookup,
    )
    adr_matrix, cached_adr_ids = _load_token_matrix(
        os.path.join(similarity_path, "adr_gda_raw_tokens.npz"),
        adr_vocabulary,
        shared_lookup,
    )
    combined = sparse.vstack((drug_matrix, adr_matrix), format="csr")#[drug_ids+adr_ids, shared_vocabulary]的稀疏矩阵
    document_frequency = np.asarray((combined > 0).sum(axis=0)).ravel()
    keep = (document_frequency >= MIN_DF) & (document_frequency <= MAX_DF * combined.shape[0])
    #只保留在 ≥2 个实体中出现过的基因 去掉在 >90% 实体中都出现的基因
    idf = np.log((1.0 + combined.shape[0]) / (1.0 + document_frequency[keep])) + 1.0#基因出现得越频繁IDF 越小；越稀有，IDF 越大
    weighted_drug = _tfidf_matrix(drug_matrix[:, keep], idf)
    weighted_adr = _tfidf_matrix(adr_matrix[:, keep], idf)
    svd_input = sparse.vstack((weighted_drug, weighted_adr), format="csr")#(drug_ids+adr_ids, 经过MIN_DF,MAX_DF筛选的基因数)的稀疏矩阵,其中只保留了topk大的基因权重
    actual_dim = min(SVD_DIM, svd_input.shape[0] - 1, svd_input.shape[1] - 1)

    print(f"[GeneSVD] 拟合共享 TF-IDF-SVD: shape={svd_input.shape}, dim={actual_dim}")
    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    combined_svd = svd.fit_transform(svd_input).astype(np.float32)#SVD降维后的矩阵[drug_ids+adr_ids, actual_dim]
    np.savez_compressed(
        cache_path,
        drug_ids=cached_drug_ids,
        adr_ids=cached_adr_ids,
        drug_svd=combined_svd[:len(drug_ids)],
        adr_svd=combined_svd[len(drug_ids):],
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump({"cache_key": cache_key}, handle, ensure_ascii=False, indent=2)
    print(f"[GeneSVD] 缓存已保存: {cache_path}")
    return {
        "drug_svd": combined_svd[:len(drug_ids)],
        "adr_svd": combined_svd[len(drug_ids):],
    }
