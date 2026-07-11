"""RPU-DGAPred E3 的 fold-safe 样本构造工具。"""

import numpy as np


def _normalize_similarity(sim):
    """把不同来源的相似度矩阵统一压到 [0, 1]，避免某个特征源数值尺度过大。"""
    sim = np.nan_to_num(np.asarray(sim, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sim_min = float(sim.min())
    sim_max = float(sim.max())
    if sim_min < 0.0 or sim_max > 1.0:
        sim = (sim - sim_min) / (sim_max - sim_min + 1e-12)
    return np.clip(sim, 0.0, 1.0)


def _fuse_similarity(features):
    """融合当前启用的多源相似度，E3 风险评分直接复用 DGAPred 的强特征组合。"""
    normalized = [_normalize_similarity(feature) for feature in features]
    return np.mean(normalized, axis=0).astype(np.float32)


def _normalize_feature_name(name):
    """统一特征名写法，便于按视图挑选矩阵。"""
    return str(name).strip().upper()


def _select_named_features(features, feature_names, selected_names):
    """按特征名取出当前视图需要的相似度矩阵。"""
    if selected_names is None:
        return list(features)
    selected = {_normalize_feature_name(name) for name in selected_names}
    return [
        feature
        for feature, name in zip(features, feature_names)
        if _normalize_feature_name(name) in selected
    ]


def _build_visible_positive_matrix(shape, train_positive_samples):
    """只用当前训练折正样本构造可见阳性矩阵，验证/测试标签不参与风险评分。"""
    visible = np.zeros(shape, dtype=np.float32)
    if len(train_positive_samples) == 0:
        return visible
    train_positive_samples = np.asarray(train_positive_samples)
    drug_idx = train_positive_samples[:, 0].astype(int)
    side_idx = train_positive_samples[:, 1].astype(int)
    rating = train_positive_samples[:, 2].astype(np.float32)
    visible[drug_idx, side_idx] = rating
    return visible


def _hidden_pair_mask(shape, hidden_data):
    """把验证集和测试集 pair 从未观察候选池排除，训练阶段不触碰评估 pair。"""
    mask = np.zeros(shape, dtype=bool)
    if hidden_data is None or len(hidden_data) == 0:
        return mask
    hidden = np.asarray(hidden_data)
    mask[hidden[:, 0].astype(int), hidden[:, 1].astype(int)] = True
    return mask


def compute_unobserved_risk(negative_samples, visible_positive_matrix, drug_features, side_features, alpha):
    """计算未观察 pair 的潜在阳性风险，风险越高越不适合作为强负样本。"""
    negative_samples = np.asarray(negative_samples)
    drug_ids = negative_samples[:, 0].astype(int)
    side_ids = negative_samples[:, 1].astype(int)
    drug_sim = _fuse_similarity(drug_features)
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

    risk = alpha * drug_risk + (1.0 - alpha) * adr_risk
    return np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _compute_view_risk(candidate_samples, visible_positive_matrix, drug_features, side_features, alpha):
    """计算单个视图下未知 pair 的潜在阳性分数，允许视图只包含药物或 ADR 一侧。"""
    candidate_samples = np.asarray(candidate_samples)
    drug_ids = candidate_samples[:, 0].astype(int)
    side_ids = candidate_samples[:, 1].astype(int)

    has_drug_view = len(drug_features) > 0
    has_side_view = len(side_features) > 0
    drug_risk = np.zeros(len(candidate_samples), dtype=np.float32)
    adr_risk = np.zeros(len(candidate_samples), dtype=np.float32)

    if has_drug_view:
        drug_sim = _fuse_similarity(drug_features)
        # drug-side 风险：同一 ADR 下，候选 drug 是否像训练集中已知会导致该 ADR 的药物。
        for side_idx in np.unique(side_ids):
            positive_drugs = np.flatnonzero(visible_positive_matrix[:, side_idx] > 0)
            if len(positive_drugs) == 0:
                continue
            sample_idx = np.flatnonzero(side_ids == side_idx)
            drug_risk[sample_idx] = drug_sim[drug_ids[sample_idx]][:, positive_drugs].max(axis=1)

    if has_side_view:
        side_sim = _fuse_similarity(side_features)
        # side-drug 风险：同一 drug 下，候选 ADR 是否像训练集中该药物已知 ADR。
        for drug_idx in np.unique(drug_ids):
            positive_sides = np.flatnonzero(visible_positive_matrix[drug_idx, :] > 0)
            if len(positive_sides) == 0:
                continue
            sample_idx = np.flatnonzero(drug_ids == drug_idx)
            adr_risk[sample_idx] = side_sim[side_ids[sample_idx]][:, positive_sides].max(axis=1)

    if has_drug_view and has_side_view:
        risk = alpha * drug_risk + (1.0 - alpha) * adr_risk
    elif has_drug_view:
        risk = drug_risk
    else:
        risk = adr_risk
    return np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _build_consensus_views(drug_features, side_features, drug_feature_names, side_feature_names):
    """按业务含义构造多视图，缺失的特征会自动跳过该视图中的对应部分。"""
    view_defs = [
        ("结构视图", ("DGEN", "DSA"), ("GDA", "DSA")),
        ("化学视图", ("CS", "MORGAN", "MACCS", "CHEMBERTA", "UNI-MOL"), ()),
        ("语义视图", ("CHEMBERTA",), ("MESH", "GLOVE")),
        ("全特征视图", None, None),
    ]

    views = []
    for view_name, drug_names, side_names in view_defs:
        view_drug_features = _select_named_features(drug_features, drug_feature_names, drug_names)
        view_side_features = _select_named_features(side_features, side_feature_names, side_names)
        if len(view_drug_features) == 0 and len(view_side_features) == 0:
            continue
        views.append((view_name, view_drug_features, view_side_features))
    return views


def build_consensus_pseudo_positive_samples(
        candidate_negative,
        visible_positive_matrix,
        drug_features,
        side_features,
        drug_feature_names,
        side_feature_names,
        positive_count,
        args):
    """从未知 pair 中筛选多视图一致高分样本，作为弱伪阳性加入训练。

    伪阳性只参与分类分支：rating 固定为 0，避免把未知强度误当真实强度训练回归分支。
    """
    empty_train = np.zeros((0, 5), dtype=np.float32)
    empty_idx = np.zeros(0, dtype=int)
    if not getattr(args, "rpu_use_consensus_pseudo", False):
        return empty_train, empty_idx, {}

    if getattr(args, "rpu_consensus_pseudo_count", 0) > 0:
        pseudo_count = int(args.rpu_consensus_pseudo_count)
    else:
        pseudo_count = int(round(positive_count * float(args.rpu_consensus_pseudo_ratio)))
    pseudo_count = min(max(pseudo_count, 0), len(candidate_negative))
    if pseudo_count == 0:
        return empty_train, empty_idx, {}

    views = _build_consensus_views(drug_features, side_features, drug_feature_names, side_feature_names)
    if len(views) == 0:
        return empty_train, empty_idx, {}

    view_names = []
    view_scores = []
    for view_name, view_drug_features, view_side_features in views:
        view_names.append(view_name)
        view_scores.append(_compute_view_risk(
            candidate_negative,
            visible_positive_matrix,
            view_drug_features,
            view_side_features,
            alpha=args.rpu_drug_risk_weight,
        ))

    score_matrix = np.vstack(view_scores).astype(np.float32)
    mean_score = score_matrix.mean(axis=0)
    min_score = score_matrix.min(axis=0)
    std_score = score_matrix.std(axis=0)
    agree_threshold = float(args.rpu_consensus_threshold)
    min_agree_views = min(int(args.rpu_consensus_min_agree_views), len(view_scores))
    agree_count = (score_matrix >= agree_threshold).sum(axis=0)

    eligible = (
        (mean_score >= agree_threshold) &
        (min_score >= float(args.rpu_consensus_min_view_score)) &
        (std_score <= float(args.rpu_consensus_max_std)) &
        (agree_count >= min_agree_views)
    )
    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) == 0:
        return empty_train, empty_idx, {
            "view_names": view_names,
            "eligible_count": 0,
            "score_mean": float(mean_score.mean()),
            "score_max": float(mean_score.max()),
        }

    order = np.lexsort((
        std_score[eligible_idx],
        -min_score[eligible_idx],
        -mean_score[eligible_idx],
    ))
    selected_idx = eligible_idx[order[:pseudo_count]]
    selected_pairs = candidate_negative[selected_idx]
    selected_score = mean_score[selected_idx]
    selected_std = std_score[selected_idx]

    soft_label = np.clip(
        selected_score,
        float(args.rpu_min_pseudo_label),
        float(args.rpu_max_pseudo_label),
    ).astype(np.float32)
    confidence = selected_score * (1.0 - selected_std)
    pseudo_weight = np.clip(
        confidence,
        float(args.rpu_min_pseudo_weight),
        float(args.rpu_max_pseudo_weight),
    ).astype(np.float32)
    pseudo_train = np.column_stack((
        selected_pairs[:, 0].astype(np.float32),
        selected_pairs[:, 1].astype(np.float32),
        soft_label,
        np.zeros(len(selected_pairs), dtype=np.float32),
        pseudo_weight,
    ))
    return pseudo_train, selected_idx.astype(int), {
        "view_names": view_names,
        "eligible_count": int(len(eligible_idx)),
        "score_mean": float(selected_score.mean()),
        "score_min": float(selected_score.min()),
        "score_max": float(selected_score.max()),
        "std_mean": float(selected_std.mean()),
        "weight_mean": float(pseudo_weight.mean()),
    }


def to_weighted_train_samples(data_train):
    """把普通三列样本转成五列样本，供非 RPU 训练路径复用同一套 DataLoader。"""
    data_train = np.asarray(data_train, dtype=np.float32)
    labels = (data_train[:, 2] > 0).astype(np.float32)
    weights = np.ones(len(data_train), dtype=np.float32)
    return np.column_stack((
        data_train[:, 0].astype(np.float32),
        data_train[:, 1].astype(np.float32),
        labels,
        data_train[:, 2].astype(np.float32),
        weights,
    ))


def build_rpu_train_samples(
        data_train,
        hidden_data,
        DAL,
        drug_features,
        side_features,
        args,
        fold,
        drug_feature_names=None,
        side_feature_names=None):
    """按 E3 all_weighted 策略为当前 fold 构造训练样本。

    输出列固定为：
    drug_idx, side_idx, soft_label, rating, sample_weight
    """
    data_train = np.asarray(data_train)
    positive_samples = data_train[data_train[:, 2].astype(np.float32) > 0]
    visible_positive = _build_visible_positive_matrix(DAL.shape, positive_samples)

    candidate_mask = (np.asarray(DAL) <= 0) & (~_hidden_pair_mask(DAL.shape, hidden_data))
    candidate_negative = np.argwhere(candidate_mask)
    if len(candidate_negative) == 0:
        raise ValueError("RPU 未观察负样本候选池为空，无法构造训练集。")

    rng = np.random.default_rng(args.seed + fold)

    if drug_feature_names is None:
        drug_feature_names = [f"DRUG_{idx}" for idx in range(len(drug_features))]
    if side_feature_names is None:
        side_feature_names = [f"SIDE_{idx}" for idx in range(len(side_features))]

    pseudo_train, pseudo_idx, pseudo_info = build_consensus_pseudo_positive_samples(
        candidate_negative,
        visible_positive,
        drug_features,
        side_features,
        drug_feature_names,
        side_feature_names,
        len(positive_samples),
        args,
    )
    if len(pseudo_idx) > 0:
        keep_negative = np.ones(len(candidate_negative), dtype=bool)
        keep_negative[pseudo_idx] = False
        candidate_negative_for_sampling = candidate_negative[keep_negative]
    else:
        candidate_negative_for_sampling = candidate_negative

    negative_count = min(int(len(positive_samples) * args.rpu_negative_ratio), len(candidate_negative))
    negative_count = min(negative_count, len(candidate_negative_for_sampling))
    sampled_idx = rng.choice(len(candidate_negative_for_sampling), size=negative_count, replace=False)
    sampled_negative = candidate_negative_for_sampling[sampled_idx]

    if len(sampled_negative) > 0:
        risks = compute_unobserved_risk(
            sampled_negative,
            visible_positive,
            drug_features,
            side_features,
            alpha=args.rpu_drug_risk_weight,
        )
        negative_weight = np.clip(1.0 - risks, args.rpu_min_neg_weight, 1.0).astype(np.float32)
    else:
        risks = np.zeros(0, dtype=np.float32)
        negative_weight = np.zeros(0, dtype=np.float32)

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
    rpu_train = np.vstack((positive_train, pseudo_train, negative_train))
    rng.shuffle(rpu_train)

    print("[RPU-E3] all_weighted negative sampling enabled")
    print(f"[RPU-E3] train positives: {len(positive_samples)}, pseudo positives: {len(pseudo_train)}, sampled negatives: {len(sampled_negative)}")
    print(f"[RPU-E3] candidate negatives: {len(candidate_negative)}, negative_ratio: {args.rpu_negative_ratio}")
    if len(risks) > 0:
        print(f"[RPU-E3] risk mean/max: {float(risks.mean()):.4f} / {float(risks.max()):.4f}")
        print(f"[RPU-E3] negative weight mean/min: {float(negative_weight.mean()):.4f} / {float(negative_weight.min()):.4f}")
    else:
        print("[RPU-E3] risk mean/max: 0.0000 / 0.0000")
        print("[RPU-E3] negative weight mean/min: 0.0000 / 0.0000")
    if getattr(args, "rpu_use_consensus_pseudo", False):
        print(f"[RPU-Pseudo] views: {', '.join(pseudo_info.get('view_names', []))}")
        print(f"[RPU-Pseudo] eligible/select: {pseudo_info.get('eligible_count', 0)} / {len(pseudo_train)}")
        if len(pseudo_train) > 0:
            print(
                "[RPU-Pseudo] score mean/min/max: "
                f"{pseudo_info['score_mean']:.4f} / {pseudo_info['score_min']:.4f} / {pseudo_info['score_max']:.4f}"
            )
            print(
                "[RPU-Pseudo] std mean, weight mean: "
                f"{pseudo_info['std_mean']:.4f}, {pseudo_info['weight_mean']:.4f}"
            )
    return rpu_train
