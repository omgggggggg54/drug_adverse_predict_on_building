"""训练特征缓存生成工具。

这里集中处理从 ``2drug-2side/DGAPred/data`` 初始数据生成训练缓存的逻辑。
``main.py`` 只负责训练入口，不再直接夹杂特征生成细节。
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.data_utils import (
    Convert_triplelist2matrix,
    data_feature,
    get_MACCS_Similarity,
    get_MESH_Similarity,
    get_Morgan_Similarity,
    get_RDKit_Similarity,
    jaccard_similarity,
)


REQUIRED_FEATURE_FILES = [
    "drug_side.csv",
    "drug_maccs.csv",
    "drug_rdkit.csv",
    "drug_morgan.csv",
    "drug_DGen_sim.csv",
    "drug_ge_sim.csv",
    "side_mesh_sim.csv",
    "adr_GDisease_sim.csv",
]

DRUG_FEATURE_FILES = [
    "drug_maccs.csv",
    "drug_rdkit.csv",
    "drug_morgan.csv",
    "drug_DGen_sim.csv",
    "drug_ge_sim.csv",
]

ADR_FEATURE_FILES = [
    "side_mesh_sim.csv",
    "adr_GDisease_sim.csv",
]


def normalize_dir_path(path):
    """规范化目录参数，并保留旧代码依赖的末尾分隔符。"""
    normalized = os.path.normpath(path)
    return normalized + os.sep


def csv_path(base_path, filename):
    """拼接 csv 路径。"""
    return os.path.join(base_path, filename)


def save_matrix(matrix, output_path, index=None, columns=None):
    """保存矩阵，保留行列名方便核对训练顺序。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(matrix, index=index, columns=columns).to_csv(output_path, header=True, index=True)
    print(f"[FeatureCache] saved: {output_path}")


def is_legacy_range(labels, size):
    """判断旧缓存是否是 0..n-1 的数字行列名。"""
    return [str(item) for item in labels] == [str(i) for i in range(size)]


def validate_or_relabel_square_feature(path, expected_ids, label):
    """确保相似度矩阵顺序与 drug_side 行/列一致。"""
    expected_ids = [str(item) for item in expected_ids]
    feature = pd.read_csv(path, header=0, index_col=0)
    if feature.shape != (len(expected_ids), len(expected_ids)):
        raise ValueError(
            f"{label} 形状不匹配: {path}, got={feature.shape}, "
            f"expected={(len(expected_ids), len(expected_ids))}"
        )

    row_ids = [str(item) for item in feature.index]
    col_ids = [str(item) for item in feature.columns]
    if row_ids == expected_ids and col_ids == expected_ids:
        return

    if is_legacy_range(row_ids, len(expected_ids)) and is_legacy_range(col_ids, len(expected_ids)):
        save_matrix(feature.values, path, index=expected_ids, columns=expected_ids)
        print(f"[FeatureCache] relabeled legacy numeric cache: {path}")
        return

    raise ValueError(
        f"{label} 行列顺序与 drug_side 不一致: {path}。"
        f"示例 row={row_ids[:3]}, expected={expected_ids[:3]}"
    )


def validate_feature_cache_order(similarity_path, drug_ids, adr_ids):
    """校验所有缓存矩阵都严格对齐 drug_side 的行列顺序。"""
    for filename in DRUG_FEATURE_FILES:
        validate_or_relabel_square_feature(
            csv_path(similarity_path, filename),
            drug_ids,
            f"drug feature {filename}",
        )
    for filename in ADR_FEATURE_FILES:
        validate_or_relabel_square_feature(
            csv_path(similarity_path, filename),
            adr_ids,
            f"ADR feature {filename}",
        )


def read_ordered_square_feature(path, expected_ids, label):
    """读取相似度矩阵，并确认行列顺序就是当前训练矩阵顺序。"""
    expected_ids = [str(item) for item in expected_ids]
    feature = pd.read_csv(path, header=0, index_col=0)

    row_ids = [str(item) for item in feature.index]
    col_ids = [str(item) for item in feature.columns]
    expected_shape = (len(expected_ids), len(expected_ids))
    if feature.shape != expected_shape or row_ids != expected_ids or col_ids != expected_ids:
        raise ValueError(
            f"{label} 与 drug_side 顺序不一致: {path}, "
            f"got_shape={feature.shape}, expected_shape={expected_shape}, "
            f"row示例={row_ids[:3]}, expected示例={expected_ids[:3]}"
        )
    return feature.values


def build_drug_side_matrix(rawpath):
    """按 DGANet 产生训练数据脚本生成 drug-ADR 标签矩阵。"""
    drug_col, adr_col = "pert_id", "MESH_ID"

    pd_label = pd.read_csv(csv_path(rawpath, "sider_pert_mesh_list.csv"), header=0, delimiter="\t")
    pd_lincs_druglist = pd.read_csv(csv_path(rawpath, "lincs_druglist_ge_go_521.csv"), header=0)
    screen_drug_list = pd_lincs_druglist[drug_col].values
    drug_side = data_feature(pd_label, screen_list=screen_drug_list, screen_col=drug_col, del_screen_col=False)

    pd_dgen = pd.read_csv(csv_path(rawpath, "ctd_chem_pert_gene_ixns_list.csv"), header=0, delimiter="\t")
    drug_set_dgen = set(pd_dgen[drug_col].values)
    drug_side = drug_side[drug_side[drug_col].isin(drug_set_dgen)].reset_index(drop=True)

    pd_agen = pd.read_csv(csv_path(rawpath, "ctd_gene_adr_asso_list_4386.csv"), header=0, delimiter="\t")
    adr_set_gen = set(pd_agen[adr_col].values)
    drug_side = drug_side[drug_side[adr_col].isin(adr_set_gen)].reset_index(drop=True)

    return Convert_triplelist2matrix(drug_side, [drug_col, adr_col, "label"], fillna_val=0)


def generate_drug_features(rawpath, similarity_path, drug_ids, missing_files):
    """按缺失清单生成 drug 侧特征缓存。"""
    drug_ids = list(drug_ids)

    chem_files = {"drug_maccs.csv", "drug_rdkit.csv", "drug_morgan.csv"}
    if chem_files & missing_files:
        pd_cs = pd.read_csv(csv_path(rawpath, "drug_pert_similes_list.csv"), header=0, delimiter="\t")
        drug_smiles = data_feature(pd_cs, screen_list=drug_ids, screen_col="pert_id", del_screen_col=False)
        if list(drug_smiles["pert_id"].astype(str)) != drug_ids:
            raise ValueError("drug_pert_similes_list.csv 筛选后顺序与 drug_side 行顺序不一致")

        if "drug_maccs.csv" in missing_files:
            sim = get_MACCS_Similarity(drug_smiles)
            save_matrix(sim, csv_path(similarity_path, "drug_maccs.csv"), index=drug_ids, columns=drug_ids)

        if "drug_rdkit.csv" in missing_files:
            sim = get_RDKit_Similarity(drug_smiles)
            save_matrix(sim, csv_path(similarity_path, "drug_rdkit.csv"), index=drug_ids, columns=drug_ids)

        if "drug_morgan.csv" in missing_files:
            sim = get_Morgan_Similarity(drug_smiles)
            save_matrix(sim, csv_path(similarity_path, "drug_morgan.csv"), index=drug_ids, columns=drug_ids)

    if "drug_DGen_sim.csv" in missing_files:
        pd_dgen = pd.read_csv(csv_path(rawpath, "ctd_chem_pert_gene_ixns_list.csv"), header=0, delimiter="\t")
        drug_dgen = data_feature(pd_dgen, screen_list=drug_ids, screen_col="pert_id", del_screen_col=False)
        drug_dgen = drug_dgen.drop_duplicates(subset=["pert_id", "GeneSymbol"], keep="first")
        drug_dgen["ixn"] = 1
        drug_dgen = Convert_triplelist2matrix(drug_dgen, ["pert_id", "GeneSymbol", "ixn"], fillna_val=0)
        drug_dgen = drug_dgen.reindex(drug_ids, fill_value=0)
        sim = jaccard_similarity(drug_dgen)
        save_matrix(sim, csv_path(similarity_path, "drug_DGen_sim.csv"), index=drug_ids, columns=drug_ids)

    if "drug_ge_sim.csv" in missing_files:
        pd_ge = pd.read_csv(csv_path(rawpath, "LINCS_Gene_Experssion_signatures_CD.csv"), header=0)
        drug_ge = data_feature(pd_ge, screen_list=drug_ids, screen_col="pert_id", del_screen_col=False)
        if list(drug_ge["pert_id"].astype(str)) != drug_ids:
            raise ValueError("LINCS_Gene_Experssion_signatures_CD.csv 筛选后顺序与 drug_side 行顺序不一致")
        drug_ge = drug_ge.drop(columns=["pert_id"])
        sim = cosine_similarity(drug_ge)
        save_matrix(sim, csv_path(similarity_path, "drug_ge_sim.csv"), index=drug_ids, columns=drug_ids)


def generate_adr_features(rawpath, similarity_path, adr_ids, missing_files):
    """按缺失清单生成 ADR 侧特征缓存。"""
    adr_ids = list(adr_ids)

    if "side_mesh_sim.csv" in missing_files:
        adr_list = pd.DataFrame(adr_ids, columns=["MESH_ID"])
        adr_list_id = adr_list["MESH_ID"].str.replace("MESH:", "", regex=False)
        pd_label = pd.read_csv(csv_path(rawpath, "se_mesh_dict_list.csv"), header=0, delimiter="\t")
        side_mesh = data_feature(pd_label, screen_list=adr_list_id.values, screen_col="MESH_ID", del_screen_col=False)
        side_mesh_ids = ("MESH:" + side_mesh["MESH_ID"].astype(str)).tolist()
        if side_mesh_ids != adr_ids:
            raise ValueError("se_mesh_dict_list.csv 筛选后顺序与 drug_side 列顺序不一致")
        sim = get_MESH_Similarity(side_mesh)
        save_matrix(sim, csv_path(similarity_path, "side_mesh_sim.csv"), index=adr_ids, columns=adr_ids)

    if "adr_GDisease_sim.csv" in missing_files:
        pd_gdisease = pd.read_csv(csv_path(rawpath, "ctd_gene_adr_asso_list_4386.csv"), header=0, delimiter="\t")
        adr_gdisease = data_feature(pd_gdisease, screen_list=adr_ids, screen_col="MESH_ID", del_screen_col=False)
        adr_gdisease = Convert_triplelist2matrix(adr_gdisease, ["MESH_ID", "GeneSymbol", "ixn"], fillna_val=0)
        adr_gdisease = adr_gdisease.reindex(adr_ids, fill_value=0)
        sim = jaccard_similarity(adr_gdisease)
        save_matrix(sim, csv_path(similarity_path, "adr_GDisease_sim.csv"), index=adr_ids, columns=adr_ids)


def ensure_training_feature_cache(rawpath, similarity_path):
    """训练前检查缓存文件，缺失时才生成。"""
    rawpath = normalize_dir_path(rawpath)
    similarity_path = normalize_dir_path(similarity_path)
    os.makedirs(similarity_path, exist_ok=True)

    missing_files = {
        filename
        for filename in REQUIRED_FEATURE_FILES
        if not os.path.exists(csv_path(similarity_path, filename))
    }

    drug_side_path = csv_path(similarity_path, "drug_side.csv")
    if "drug_side.csv" in missing_files:
        print("[FeatureCache] 缺少 drug_side.csv，将从初始数据生成。")
        drug_side = build_drug_side_matrix(rawpath)
        save_matrix(drug_side, drug_side_path, index=drug_side.index, columns=drug_side.columns)
    else:
        drug_side = pd.read_csv(drug_side_path, header=0, index_col=0)

    if missing_files - {"drug_side.csv"}:
        print("[FeatureCache] 发现缺失缓存，将从初始数据补齐:")
        for filename in REQUIRED_FEATURE_FILES:
            status = "missing" if filename in missing_files else "exists "
            print(f"  - {status}: {filename}")
        generate_drug_features(rawpath, similarity_path, drug_side.index, missing_files)
        generate_adr_features(rawpath, similarity_path, drug_side.columns, missing_files)
    else:
        print("[FeatureCache] pythonPredict 下训练缓存已齐全，检查行列顺序。")

    validate_feature_cache_order(similarity_path, drug_side.index, drug_side.columns)
    print("[FeatureCache] 缓存检查/补齐完成，特征顺序已对齐 drug_side。")
    return rawpath, similarity_path
