"""RPU-DGAPred E3 的 fold-safe 样本构造工具。"""

import numpy as np


# 当前最佳主线的固定 RPU 配置，不再暴露为命令行调参项。
NEGATIVE_RATIO = 10
RANDOM_SEED = 42
MIN_NEG_WEIGHT = 0.2
DRUG_RISK_WEIGHT = 0.7


def _normalize_similarity(sim):
    """把不同来源的相似度矩阵统一压到 [0, 1]，避免某个特征源数值尺度过大。"""
    sim = np.asarray(sim, dtype=np.float32)
    sim_min = float(sim.min())
    sim_max = float(sim.max())
    if sim_min < 0.0 or sim_max > 1.0:
        sim = (sim - sim_min) / (sim_max - sim_min + 1e-12)
    return np.clip(sim, 0.0, 1.0)


def _fuse_similarity(features):
    """融合当前启用的多源相似度，E3 风险评分直接复用 DGAPred 的强特征组合。"""
    normalized = [_normalize_similarity(feature) for feature in features]
    return np.mean(normalized, axis=0).astype(np.float32)


def _build_visible_positive_matrix(shape, train_positive_samples):
    """只用当前训练折正样本构造可见阳性矩阵，验证/测试标签不参与风险评分。"""
    visible = np.zeros(shape, dtype=np.float32)
    train_positive_samples = np.asarray(train_positive_samples)
    drug_idx = train_positive_samples[:, 0].astype(int)
    side_idx = train_positive_samples[:, 1].astype(int)
    rating = train_positive_samples[:, 2].astype(np.float32)
    visible[drug_idx, side_idx] = rating
    return visible


def _hidden_pair_mask(shape, hidden_data):
    """把验证集和测试集 pair 从未观察候选池排除，训练阶段不触碰评估 pair。"""
    mask = np.zeros(shape, dtype=bool)
    hidden = np.asarray(hidden_data)
    mask[hidden[:, 0].astype(int), hidden[:, 1].astype(int)] = True
    return mask


def compute_unobserved_risk(negative_samples, visible_positive_matrix, drug_features, side_features, alpha):
    """计算未观察 pair 的潜在阳性风险，风险越高越不适合作为强负样本。"""
    negative_samples = np.asarray(negative_samples)
    drug_ids = negative_samples[:, 0].astype(int)
    side_ids = negative_samples[:, 1].astype(int)
    drug_sim = _fuse_similarity(drug_features)#normalized and mean
    side_sim = _fuse_similarity(side_features)

    drug_risk = np.zeros(len(negative_samples), dtype=np.float32)
    adr_risk = np.zeros(len(negative_samples), dtype=np.float32)

    # drug-side 风险：同一 ADR 下，候选 drug 与训练可见阳性 drug 的最大相似度。
    for side_idx in np.unique(side_ids):
        positive_drugs = np.flatnonzero(visible_positive_matrix[:, side_idx] > 0)
        if len(positive_drugs) == 0:
            continue
        sample_idx = np.flatnonzero(side_ids == side_idx)
        drug_risk[sample_idx] = drug_sim[drug_ids[sample_idx]][:, positive_drugs].max(axis=1)

    # side-drug 风险：同一 drug 下，候选 ADR 与训练可见阳性 ADR 的最大相似度。
    for drug_idx in np.unique(drug_ids):
        positive_sides = np.flatnonzero(visible_positive_matrix[drug_idx, :] > 0)
        if len(positive_sides) == 0:
            continue
        sample_idx = np.flatnonzero(drug_ids == drug_idx)
        adr_risk[sample_idx] = side_sim[side_ids[sample_idx]][:, positive_sides].max(axis=1)

    return np.clip(alpha * drug_risk + (1.0 - alpha) * adr_risk, 0.0, 1.0)


def to_weighted_train_samples(data_train):
    """关闭 RPU 时将三列训练样本转换为统一的五列训练格式。"""
    data_train = np.asarray(data_train, dtype=np.float32)
    labels = (data_train[:, 2] > 0).astype(np.float32)
    return np.column_stack((
        data_train[:, 0].astype(np.float32),
        data_train[:, 1].astype(np.float32),
        labels,
        data_train[:, 2].astype(np.float32),
        np.ones(len(data_train), dtype=np.float32),
    ))


def build_rpu_train_samples(
        data_train,
        hidden_data,
        DAL,#(n_drugs, n_sides)
        drug_features,
        side_features,
        seed=RANDOM_SEED):
    """按固定 RPU 负采样和 risk 权重为当前 fold 构造训练样本。

    输出列固定为：
    drug_idx, side_idx, soft_label, rating, sample_weight
    """
    data_train = np.asarray(data_train)
    positive_samples = data_train[data_train[:, 2].astype(np.float32) > 0]
    visible_positive = _build_visible_positive_matrix(DAL.shape, positive_samples)

    candidate_mask = (np.asarray(DAL) <= 0) & (~_hidden_pair_mask(DAL.shape, hidden_data))#hidden_data->验证集和测试集 pair
    candidate_negative = np.argwhere(candidate_mask)#排除掉验证集和测试集 pair 后的所有未知 pair

    # 使用局部随机生成器，不污染全局 NumPy 状态；调用方始终传入统一 SEED=42。
    # 不把 fold 编号混入 seed，确保同一基础训练数据在不同运行中采样完全一致。
    rng = np.random.default_rng(seed)

    negative_count = min(int(len(positive_samples) * NEGATIVE_RATIO), len(candidate_negative))
    sampled_idx = rng.choice(len(candidate_negative), size=negative_count, replace=False)#从未知负样本中随机抽样
    sampled_negative = candidate_negative[sampled_idx]

    risks = compute_unobserved_risk(#计算剩下负样本的risk
        sampled_negative,
        visible_positive,
        drug_features,
        side_features,
        alpha=DRUG_RISK_WEIGHT,
    )
    negative_weight = np.clip(1.0 - risks, MIN_NEG_WEIGHT, 1.0).astype(np.float32)

    positive_train = np.column_stack((
        positive_samples[:, 0].astype(np.float32),
        positive_samples[:, 1].astype(np.float32),
        np.ones(len(positive_samples), dtype=np.float32),
        positive_samples[:, 2].astype(np.float32),
        np.ones(len(positive_samples), dtype=np.float32),
    ))
    negative_train = np.column_stack((
        sampled_negative[:, 0].astype(np.float32),
        sampled_negative[:, 1].astype(np.float32),
        np.zeros(len(sampled_negative), dtype=np.float32),
        np.zeros(len(sampled_negative), dtype=np.float32),
        negative_weight,
    ))
    rpu_train = np.vstack((positive_train, negative_train))#真实正样本 + 加权负样本
    rng.shuffle(rpu_train)

    print("[RPU-E3] all_weighted negative sampling enabled")
    print(f"[RPU-E3] train positives: {len(positive_samples)}, sampled negatives: {len(sampled_negative)}")
    print(f"[RPU-E3] candidate negatives: {len(candidate_negative)}, negative_ratio: {NEGATIVE_RATIO}")
    print(f"[RPU-E3] risk mean/max: {float(risks.mean()):.4f} / {float(risks.max()):.4f}")
    print(f"[RPU-E3] negative weight mean/min: {float(negative_weight.mean()):.4f} / {float(negative_weight.min()):.4f}")
    return rpu_train
