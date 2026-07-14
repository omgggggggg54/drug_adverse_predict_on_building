"""原始分子、基因和 MESH 特征缓存工具。"""

import ast
import csv
import os

import numpy as np
import pandas as pd


def _save_raw_feature(features, output_path, ids):
    """保存原始特征和实体顺序，后续读取时必须严格校验。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        ids=np.asarray([str(item) for item in ids]),
    )
    print(f"[RawFeature] saved: {output_path}")


def read_ordered_raw_feature(path, expected_ids, label):
    """读取原始特征，并确认实体顺序与训练标签矩阵一致。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} 缓存不存在: {path}")

    expected_ids = [str(item) for item in expected_ids]
    with np.load(path, allow_pickle=False) as cache:
        features = cache["features"].astype(np.float32)
        ids = [str(item) for item in cache["ids"].tolist()]

    if features.ndim != 2 or features.shape[0] != len(expected_ids) or ids != expected_ids:
        raise ValueError(
            f"{label} 与训练矩阵顺序不一致: path={path}, "
            f"shape={features.shape}, expected_rows={len(expected_ids)}"
        )
    return features


def ensure_drug_fingerprint_raw(similarity_path, drug_ids, output_filename, fingerprint_type):
    """按训练药物顺序生成 RDKit 原始二进制指纹。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    from rdkit import Chem, DataStructs

    drug_ids = [str(item) for item in drug_ids]
    mapping_path = os.path.join(similarity_path, "aligned_mapping", "training_matrix_drugs.csv")
    mapping = pd.read_csv(mapping_path).set_index("pert_id").reindex(drug_ids)
    if mapping["canonical_smiles"].isna().any():
        missing = mapping.index[mapping["canonical_smiles"].isna()].tolist()[:5]
        raise ValueError(f"{fingerprint_type} 原始指纹缺少 canonical_smiles，示例: {missing}")

    bit_count = 2048
    fingerprints = np.zeros((len(mapping), bit_count), dtype=np.float32)
    for index, smiles in enumerate(mapping["canonical_smiles"]):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"{fingerprint_type} 无法解析 SMILES: {smiles}")
        if fingerprint_type == "rdkit":
            fingerprint = Chem.RDKFingerprint(molecule, fpSize=bit_count)
        else:
            raise ValueError(f"不支持的指纹类型: {fingerprint_type}")
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprints[index])

    _save_raw_feature(fingerprints, output_path, drug_ids)
    return output_path


def _save_token_feature(token_lists, output_path, ids):
    """保存变长 token 序列，避免稀疏基因特征展开成大矩阵。"""
    vocabulary = sorted({token for tokens in token_lists for token in tokens})
    token_to_id = {token: index + 1 for index, token in enumerate(vocabulary)}
    offsets = np.zeros(len(token_lists) + 1, dtype=np.int64)
    encoded_lists = []
    for index, tokens in enumerate(token_lists):
        encoded = np.asarray([token_to_id[token] for token in sorted(tokens)], dtype=np.int32)
        encoded_lists.append(encoded)
        offsets[index + 1] = offsets[index] + len(encoded)
    token_ids = np.concatenate(encoded_lists) if encoded_lists else np.zeros(0, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        token_ids=token_ids,
        offsets=offsets,
        ids=np.asarray([str(item) for item in ids]),
        vocab_size=np.asarray(len(vocabulary) + 1, dtype=np.int64),
    )
    print(
        f"[RawFeature] saved token cache: {output_path}, "
        f"entities={len(ids)}, tokens={len(token_ids)}, vocab={len(vocabulary)}"
    )


def read_ordered_token_feature(path, expected_ids, label):
    """读取 token 缓存，并校验实体顺序和 offsets 完整性。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} token 缓存不存在: {path}")

    expected_ids = [str(item) for item in expected_ids]
    with np.load(path, allow_pickle=False) as cache:
        token_ids = cache["token_ids"].astype(np.int64)
        offsets = cache["offsets"].astype(np.int64)
        ids = [str(item) for item in cache["ids"].tolist()]
        vocab_size = int(cache["vocab_size"])

    if ids != expected_ids or len(offsets) != len(expected_ids) + 1:
        raise ValueError(f"{label} token 缓存与训练矩阵顺序不一致: {path}")
    if offsets[0] != 0 or offsets[-1] != len(token_ids) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{label} token offsets 非法: {path}")
    return {
        "token_ids": token_ids,
        "offsets": offsets,
        "vocab_size": vocab_size,
        "name": label,
    }


def _ensure_gene_token_feature(rawpath, similarity_path, entity_ids, output_filename, source_filename, entity_column):
    """从基因关联表流式提取指定实体的去重 GeneSymbol token。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    entity_ids = [str(item) for item in entity_ids]
    entity_sets = {entity_id: set() for entity_id in entity_ids}
    source_path = os.path.join(rawpath, source_filename)
    with open(source_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entity_id = str(row[entity_column])
            if entity_id in entity_sets:
                entity_sets[entity_id].add(str(row["GeneSymbol"]))

    _save_token_feature([entity_sets[entity_id] for entity_id in entity_ids], output_path, entity_ids)
    return output_path


def ensure_drug_gene_raw(rawpath, similarity_path, drug_ids, output_filename):
    """按训练药物顺序生成 DGen 原始 gene token 缓存。"""
    return _ensure_gene_token_feature(
        rawpath,
        similarity_path,
        drug_ids,
        output_filename,
        "ctd_chem_pert_gene_ixns_list.csv",
        "pert_id",
    )


def ensure_adr_gene_raw(rawpath, similarity_path, adr_ids, output_filename):
    """按训练 ADR 顺序生成 GDA 原始 gene token 缓存。"""
    return _ensure_gene_token_feature(
        rawpath,
        similarity_path,
        adr_ids,
        output_filename,
        "ctd_gene_adr_asso_list_4386.csv",
        "MESH_ID",
    )


def ensure_adr_mesh_raw(rawpath, similarity_path, adr_ids, output_filename):
    """按训练 ADR 顺序生成 MESH 本体祖先节点 token 缓存。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    adr_ids = [str(item) for item in adr_ids]
    node_sets = {adr_id: set() for adr_id in adr_ids}
    source_path = os.path.join(rawpath, "se_mesh_dict_list.csv")
    with open(source_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            adr_id = f"MESH:{row['MESH_ID']}"
            if adr_id in node_sets:
                node_sets[adr_id].update(str(node) for node in ast.literal_eval(row["Dict_MESH"]).keys())

    _save_token_feature([node_sets[adr_id] for adr_id in adr_ids], output_path, adr_ids)
    return output_path
