"""本地语义/指纹特征生成工具。

这里统一处理几类可选特征：
1. 基于 HuggingFace 本地模型的文本或 SMILES 编码特征
2. 基于 Uni-Mol 的药物表示特征
3. 基于 GloVe 的 ADR 文本平均词向量特征
"""

import os
import re

import numpy as np
import pandas as pd


def _save_matrix(matrix, output_path, ids):
    """保存相似度矩阵，行列名严格使用训练矩阵顺序。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(matrix, index=ids, columns=ids).to_csv(output_path, header=True, index=True)
    print(f"[EmbeddingFeature] saved: {output_path}")


def _load_transformer_runtime(feature_name):
    """延迟导入大模型依赖，未启用语义特征时不影响基线训练。"""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            f"启用 {feature_name} 特征需要安装 torch 和 transformers。"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _resolve_model_path(model_path, feature_name):
    """确认本地模型目录存在，禁止训练时联网下载。"""
    model_path = os.path.abspath(model_path)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"{feature_name} 本地模型目录不存在: {model_path}。"
            f"请下载 HuggingFace 模型到本地，或用对应 *_model_path 参数指定路径。"
        )
    return model_path


def _resolve_file_path(file_path, feature_name):
    """确认本地文件存在。"""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{feature_name} 文件不存在: {file_path}")
    return file_path


def _mean_pool(last_hidden_state, attention_mask):
    """按 attention mask 做 mean pooling，避免 padding token 影响向量。"""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_texts(texts, model_path, feature_name, batch_size=32):
    """用本地 HuggingFace 模型编码文本或 SMILES。"""
    torch, AutoModel, AutoTokenizer = _load_transformer_runtime(feature_name)
    model_path = _resolve_model_path(model_path, feature_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    max_length = getattr(tokenizer, "model_max_length", 512)
    if max_length is None or max_length > 4096:
        max_length = 512

    embeddings = []
    texts = [str(item) for item in texts]
    print(f"[EmbeddingFeature] encoding {feature_name}: {len(texts)} items on {device}")
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            embeddings.append(pooled.detach().cpu().numpy())
            print(f"[EmbeddingFeature] {feature_name}: {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(embeddings).astype(np.float32)


def _embedding_to_similarity(embeddings):
    """把 embedding 转成 0~1 cosine 相似度矩阵。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    sim = embeddings @ embeddings.T
    sim = (sim + 1.0) * 0.5
    sim = np.clip(sim, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    return sim


def _load_drug_mapping(similarity_path, drug_ids, feature_name):
    """按训练矩阵顺序读取药物映射表。"""
    drug_ids = [str(item) for item in drug_ids]
    mapping_path = os.path.join(similarity_path, "aligned_mapping", "training_matrix_drugs.csv")
    mapping = pd.read_csv(mapping_path)
    mapping["pert_id"] = mapping["pert_id"].astype(str)
    mapping = mapping.set_index("pert_id").reindex(drug_ids)
    if mapping.index.isna().any():
        raise ValueError(f"{feature_name} 药物映射表重排失败。")
    return mapping


def _load_adr_mapping(similarity_path, adr_ids, feature_name):
    """按训练矩阵顺序读取 ADR 映射表。"""
    adr_ids = [str(item) for item in adr_ids]
    mapping_path = os.path.join(similarity_path, "aligned_mapping", "adr_mesh_cui_medra.csv")
    mapping = pd.read_csv(mapping_path)
    mapping["MESH"] = mapping["MESH"].astype(str)
    mapping = mapping.set_index("MESH").reindex(adr_ids)
    if mapping.index.isna().any():
        raise ValueError(f"{feature_name} ADR 映射表重排失败。")
    return mapping


def _load_unimol_runtime(weight_dir, feature_name):
    """延迟导入 Uni-Mol，并指定本地权重目录，避免运行时联网下载。"""
    weight_dir = os.path.abspath(weight_dir)
    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(f"{feature_name} 本地权重目录不存在: {weight_dir}")
    required_files = ["mol_pre_no_h_220816.pt", "mol.dict.txt"]
    missing = [name for name in required_files if not os.path.exists(os.path.join(weight_dir, name))]
    if missing:
        raise FileNotFoundError(f"{feature_name} 权重目录缺少文件: {missing}")

    os.environ["UNIMOL_WEIGHT_DIR"] = weight_dir
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError(f"启用 {feature_name} 特征需要安装 unimol-tools。") from exc
    return UniMolRepr


def _tokenize_glove_text(text):
    """把 ADR 文本切成适合 GloVe 查表的 token。"""
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _load_glove_vectors(glove_path):
    """读取 GloVe 词向量文本文件。"""
    glove_path = _resolve_file_path(glove_path, "GloVe")
    vectors = {}
    dim = None
    with open(glove_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) <= 2:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim is None:
                dim = vec.shape[0]
            vectors[word] = vec
    if not vectors or dim is None:
        raise ValueError(f"GloVe 文件内容为空或格式不正确: {glove_path}")
    return vectors, dim


def _encode_glove_texts(texts, glove_path, feature_name):
    """对 ADR 文本做平均词向量编码。"""
    glove_vectors, dim = _load_glove_vectors(glove_path)
    embeddings = []
    print(f"[EmbeddingFeature] encoding {feature_name}: {len(texts)} items with GloVe")
    for text in texts:
        vectors = [glove_vectors[token] for token in _tokenize_glove_text(text) if token in glove_vectors]
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(dim, dtype=np.float32))
    return np.vstack(embeddings).astype(np.float32)


def _encode_unimol_smiles(smiles_list, weight_dir, feature_name, batch_size=16):
    """用 Uni-Mol 官方工具提取分子级 CLS 表示。"""
    UniMolRepr = _load_unimol_runtime(weight_dir, feature_name)
    print(f"[EmbeddingFeature] encoding {feature_name}: {len(smiles_list)} items with Uni-Mol")
    encoder = UniMolRepr(
        data_type="molecule",
        remove_hs=True,
        batch_size=batch_size,
        use_cuda=True,
    )
    output = encoder.get_repr([str(item) for item in smiles_list], return_atomic_reprs=False)
    if isinstance(output, dict):
        output = output["cls_repr"]
    return np.asarray(output, dtype=np.float32)


def ensure_drug_embedding_similarity(similarity_path, drug_ids, model_path, output_filename, feature_name):
    """按 drug_side 行顺序生成药物 SMILES 语义相似度缓存。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    drug_ids = [str(item) for item in drug_ids]
    mapping = _load_drug_mapping(similarity_path, drug_ids, feature_name)
    if mapping["canonical_smiles"].isna().any():
        missing = mapping.index[mapping["canonical_smiles"].isna()].tolist()[:5]
        raise ValueError(f"{feature_name} 缺少 canonical_smiles，示例: {missing}")

    embeddings = _encode_texts(mapping["canonical_smiles"].tolist(), model_path, feature_name)
    sim = _embedding_to_similarity(embeddings)
    _save_matrix(sim, output_path, drug_ids)
    return output_path


def ensure_drug_unimol_similarity(similarity_path, drug_ids, weight_dir, output_filename, feature_name):
    """按训练药物顺序生成 Uni-Mol 相似度缓存。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    drug_ids = [str(item) for item in drug_ids]
    mapping = _load_drug_mapping(similarity_path, drug_ids, feature_name)
    if mapping["canonical_smiles"].isna().any():
        missing = mapping.index[mapping["canonical_smiles"].isna()].tolist()[:5]
        raise ValueError(f"{feature_name} 缺少 canonical_smiles，示例: {missing}")

    embeddings = _encode_unimol_smiles(mapping["canonical_smiles"].tolist(), weight_dir, feature_name)
    sim = _embedding_to_similarity(embeddings)
    _save_matrix(sim, output_path, drug_ids)
    return output_path


def ensure_adr_glove_similarity(similarity_path, adr_ids, glove_path, output_filename, feature_name):
    """按训练 ADR 顺序生成 GloVe 语义相似度缓存。"""
    output_path = os.path.join(similarity_path, output_filename)
    if os.path.exists(output_path):
        return output_path

    adr_ids = [str(item) for item in adr_ids]
    mapping = _load_adr_mapping(similarity_path, adr_ids, feature_name)
    if mapping["ADRNAME"].isna().any():
        missing = mapping.index[mapping["ADRNAME"].isna()].tolist()[:5]
        raise ValueError(f"{feature_name} 缺少 ADRNAME，示例: {missing}")

    embeddings = _encode_glove_texts(mapping["ADRNAME"].tolist(), glove_path, feature_name)
    sim = _embedding_to_similarity(embeddings)
    _save_matrix(sim, output_path, adr_ids)
    return output_path
