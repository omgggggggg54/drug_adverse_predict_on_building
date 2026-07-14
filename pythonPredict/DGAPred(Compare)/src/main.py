"""DGAPred Training Pipeline

Main script for training and evaluating the DGAPred model for drug-side effect prediction.
Implements 5-fold cross-validation with advanced graph neural network techniques.
"""

import os
import sys
import time
import random
import pickle
import argparse
from math import sqrt
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from model.model import DGAPred
from utils.data_utils import jaccard_similarity
from utils.feature_generation import ensure_training_feature_cache, read_ordered_square_feature
from utils.raw_feature_generation import (
    ensure_adr_gene_raw,
    ensure_adr_mesh_raw,
    ensure_drug_fingerprint_raw,
    ensure_drug_gene_raw,
    read_ordered_raw_feature,
    read_ordered_token_feature,
)
from utils.rpu_utils import MIN_NEG_WEIGHT, build_rpu_train_samples, to_weighted_train_samples
# ChemProp 依赖已移除

FUSION_REGULARIZATION = 0.01

# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def configure_cpu_threads(torch_threads, torch_interop_threads, prefer_cuda=False):
    """限制 PyTorch 的 CPU 线程池，避免 CUDA 训练时 CPU 线程池长期忙等。"""
    if prefer_cuda and torch_threads <= 0:
        torch_threads = min(4, max(1, os.cpu_count() or 1))
    if prefer_cuda and torch_interop_threads <= 0:
        torch_interop_threads = 1

    # OpenMP 在线程空闲时默认会忙等，这会让几个核心看起来一直满载。
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")

    if torch_threads > 0:
        # intra-op 线程负责单个算子内部的 CPU 并行；CUDA 训练也会用它做调度和部分 CPU 计算。
        torch.set_num_threads(torch_threads)
        os.environ["OMP_NUM_THREADS"] = str(torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(torch_threads)

    if torch_interop_threads > 0:
        # inter-op 线程负责多个算子之间的并行调度；设小一点可以减少 CPU 抢占。
        torch.set_num_interop_threads(torch_interop_threads)

    print(f"[CPU] torch_threads={torch.get_num_threads()}, "
          f"torch_interop_threads={torch.get_num_interop_threads()}")


# 设置系统路径
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_label(args):
    """加载已经准备好的 drug-side effect 标签矩阵。
    
    Args:
        args: 命令行参数
        
    Returns:
        drug_ids: drug_side 行顺序中的药物ID
        adr_ids: drug_side 列顺序中的ADR ID
        drug_side: 药物-ADR标签矩阵
    """
    # Check cache
    cache_filename = "drug_side.csv"
    cache_path = os.path.join(args.similarity_path, cache_filename)
    
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"缺少标签矩阵缓存: {cache_path}。请先运行 ensure_training_feature_cache。"
        )

    print(f"[Cache] Loading from cache: {cache_path}")
    drug_side = pd.read_csv(cache_path, header=0, index_col=0)
    return list(drug_side.index), list(drug_side.columns), drug_side 

def load_drug_feature(drug_ids, args):
    """加载药物相似性特征矩阵。
    
    特征包括:
    - 药物-基因相互作用 (DGen)
    - 基因表达 (GE) 
    - 化学结构 (CS)
    
    Args:
        drug_ids: drug_side 行顺序中的药物ID
        args: 命令行参数
        
    Returns:
        药物相似性矩阵列表
    """
    print(f"\n{'='*60}")
    print("加载药物特征")
    print(f"{'='*60}")
    
    selected = parse_feature_tokens(args.drug_features)
    feature_files = [
        ("DGen", "drug_DGen_sim.csv"),
        ("CS", "drug_rdkit.csv"),
    ]

    drug_features = []
    drug_feature_names = []
    print(f"\n药物特征已加载:")
    for name, filename in feature_files:
        if name.upper() not in selected:
            print(f"  - {name}: skipped")
            continue
        feature = read_ordered_square_feature(
            os.path.join(args.similarity_path, filename),
            drug_ids,
            f"药物特征 {name}"
        )
        drug_features.append(feature)
        drug_feature_names.append(name)
        print(f"  - {name}: {feature.shape}")
    if len(drug_features) == 0:
        raise ValueError("至少需要启用一个 drug feature")
    print(f"{'='*60}\n")
    
    return drug_features, drug_feature_names

def load_adr_feature(adr_ids, args):
    """加载ADR相似性特征矩阵。
    
    特征包括:
    - MESH本体相似度
    - 基因-疾病关联(GDA)
    
    Args:
        adr_ids: drug_side 列顺序中的ADR ID
        args: 命令行参数
        
    Returns:
        ADR相似性矩阵列表
    """
    print(f"\n{'='*60}")
    print("加载ADR特征")
    print(f"{'='*60}")
    
    selected = parse_feature_tokens(args.adr_features)
    feature_files = [
        ("MESH", "side_mesh_sim.csv"),
        ("GDA", "adr_GDisease_sim.csv"),
    ]

    side_features = []
    side_feature_names = []
    print(f"\nADR特征已加载:")
    for name, filename in feature_files:
        if name.upper() not in selected:
            print(f"  - {name}: skipped")
            continue
        feature = read_ordered_square_feature(
            os.path.join(args.similarity_path, filename),
            adr_ids,
            f"ADR特征 {name}"
        )
        side_features.append(feature)
        side_feature_names.append(name)
        print(f"  - {name}: {feature.shape}")
    if len(side_features) == 0:
        raise ValueError("至少需要启用一个 ADR feature")
    print(f"{'='*60}\n")
    
    return side_features, side_feature_names


def split_val_test(data_test, val_ratio=0.2, seed=42):
    """旧评估口径：从当前测试折里切出验证集和最终测试集。"""
    data_test = np.array(data_test)
    labels = data_test[:, 2].astype(int)
    n_splits = max(2, int(round(1.0 / val_ratio)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return data_test[val_idx].tolist(), data_test[test_idx].tolist()


def split_train_val(data_train, val_ratio=0.2, seed=42):
    """严格评估口径：从训练折里切验证集，外层测试折保持完全独立。"""
    data_train = np.array(data_train)
    labels = data_train[:, 2].astype(int)
    n_splits = max(2, int(round(1.0 / val_ratio)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return data_train[train_idx].tolist(), data_train[val_idx].tolist()


def parse_feature_tokens(text):
    """把逗号分隔的特征开关字符串转成大写集合。"""
    disabled_tokens = {"", "NONE", "NULL", "OFF", "FALSE"}
    return {
        item.strip().upper()
        for item in str(text or "").split(",")
        if item.strip().upper() not in disabled_tokens
    }


RAW_DRUG_DENSE_FEATURE_SPECS = [
    ("CS", "CS", "drug_rdkit_raw.npz", "rdkit"),
]

RAW_DRUG_TOKEN_FEATURE_SPECS = [
    ("DGEN", "DGen", "drug_dgen_raw_tokens.npz"),
]

RAW_ADR_DENSE_FEATURE_SPECS = []

RAW_ADR_TOKEN_FEATURE_SPECS = [
    ("MESH", "MESH", "adr_mesh_raw_tokens.npz"),
    ("GDA", "GDA", "adr_gda_raw_tokens.npz"),
]


def normalize_raw_configuration(args):
    """统一 raw 参数语义，none 模式强制清空所有原始特征选择。"""
    args.raw_feature_mode = str(args.raw_feature_mode or "none").strip().lower()
    if args.raw_feature_mode == "none":
        args.raw_drug_features = ""
        args.raw_adr_features = ""
        return
    args.raw_drug_features = args.raw_drug_features or "DGen,CS"
    args.raw_adr_features = args.raw_adr_features or "MESH,GDA"


def load_raw_drug_features(drug_ids, args):
    """加载可直接输入模型的药物原始表示。"""
    if args.raw_feature_mode == "none":
        return {}, {}
    selected = parse_feature_tokens(args.raw_drug_features)
    dense_features = {}
    token_features = {}
    for token, source_name, filename, fingerprint_type in RAW_DRUG_DENSE_FEATURE_SPECS:
        if token not in selected:
            continue
        output_path = os.path.join(args.similarity_path, filename)
        ensure_drug_fingerprint_raw(args.similarity_path, drug_ids, filename, fingerprint_type)
        feature = read_ordered_raw_feature(output_path, drug_ids, source_name)
        dense_features[source_name.upper()] = feature
        print(f"[RawFeature] drug {source_name}: {feature.shape}")
    for token, source_name, filename in RAW_DRUG_TOKEN_FEATURE_SPECS:
        if token not in selected:
            continue
        output_path = os.path.join(args.similarity_path, filename)
        ensure_drug_gene_raw(args.rawpath, args.similarity_path, drug_ids, filename)
        feature = read_ordered_token_feature(output_path, drug_ids, source_name)
        token_features[source_name.upper()] = feature
        print(f"[RawFeature] drug {source_name}: token_count={len(feature['token_ids'])}, "
              f"vocab={feature['vocab_size']}")
    return dense_features, token_features


def load_raw_adr_features(adr_ids, args):
    """加载可直接输入模型的 ADR 原始表示。"""
    if args.raw_feature_mode == "none":
        return {}, {}
    selected = parse_feature_tokens(args.raw_adr_features)
    dense_features = {}
    token_features = {}
    for token, source_name, filename in RAW_ADR_TOKEN_FEATURE_SPECS:
        if token not in selected:
            continue
        output_path = os.path.join(args.similarity_path, filename)
        if token == "MESH":
            ensure_adr_mesh_raw(args.rawpath, args.similarity_path, adr_ids, filename)
        else:
            ensure_adr_gene_raw(args.rawpath, args.similarity_path, adr_ids, filename)
        feature = read_ordered_token_feature(output_path, adr_ids, source_name)
        token_features[source_name.upper()] = feature
        print(f"[RawFeature] ADR {source_name}: token_count={len(feature['token_ids'])}, "
              f"vocab={feature['vocab_size']}")
    return dense_features, token_features


def raw_dense_feature_list(raw_features):
    """按缓存配置顺序返回 raw-only 模型中的 dense 特征视图。"""
    return list(raw_features.values()), list(raw_features.keys())


def add_raw_dsa_features(drug_features, side_features, drug_feature_names, side_feature_names, drug_side, hidden_data, args):
    """为 raw-only 模型加入 fold-safe 的原始 drug-ADR 关联 profile。"""
    selected_drug = parse_feature_tokens(args.drug_features)
    selected_side = parse_feature_tokens(args.adr_features)
    if "DSA" not in selected_drug and "DSA" not in selected_side:
        return drug_features, side_features, drug_feature_names, side_feature_names

    visible_profile = drug_side.values.astype(np.float32, copy=True)
    hidden_array = np.asarray(hidden_data)
    visible_profile[hidden_array[:, 0].astype(int), hidden_array[:, 1].astype(int)] = 0.0
    if "DSA" in selected_drug:
        drug_features.append(visible_profile)
        drug_feature_names.append("DSARaw")
    if "DSA" in selected_side:
        side_features.append(visible_profile.T)
        side_feature_names.append("DSARaw")
    return drug_features, side_features, drug_feature_names, side_feature_names


def build_dense_feature_matrix(features, entity_count):
    """拼接 dense 特征；纯 token 一侧保留零宽矩阵供模型统一索引。"""
    if not features:
        return np.zeros((entity_count, 0), dtype=np.float32)
    return np.hstack(features).astype(np.float32)


def move_token_features_to_device(token_features, device):
    """将原始 token 缓存一次性转为训练设备上的只读张量。"""
    if not token_features:
        return None
    return [
        (
            torch.as_tensor(feature["token_ids"], dtype=torch.long, device=device),
            torch.as_tensor(feature["offsets"], dtype=torch.long, device=device),
        )
        for feature in token_features.values()
    ]


def token_feature_entity_count(token_features):
    """从任意一个 token 特征的 offsets 推断实体数量。"""
    first_feature = next(iter(token_features.values()))
    return first_feature["offsets"].shape[0] - 1


def add_dsa_features(drug_features, side_features, drug_feature_names, side_feature_names, drug_side, hidden_data, args):
    """按 DGANet baseline 构造 DSA 特征。

    验证集和最终测试集位置会先置 0，避免 DSA 特征看到待评估标签。
    """
    selected_drug = parse_feature_tokens(args.drug_features)
    selected_side = parse_feature_tokens(args.adr_features)
    if "DSA" not in selected_drug and "DSA" not in selected_side:
        return drug_features, side_features, drug_feature_names, side_feature_names

    drug_side_for_sim = drug_side.values.copy()
    hidden_array = np.array(hidden_data)
    drug_side_for_sim[hidden_array[:, 0].astype(int), hidden_array[:, 1].astype(int)] = 0
    drug_side_sim = jaccard_similarity(drug_side_for_sim)
    side_drug_sim = jaccard_similarity(drug_side_for_sim.T)

    print(f"[DSA] drug-side similarity: {drug_side_sim.shape}")
    print(f"[DSA] side-drug similarity: {side_drug_sim.shape}")
    fold_drug_features = list(drug_features)
    fold_side_features = list(side_features)
    fold_drug_feature_names = list(drug_feature_names)
    fold_side_feature_names = list(side_feature_names)
    if "DSA" in selected_drug:
        fold_drug_features.append(drug_side_sim)
        fold_drug_feature_names.append("DSA")
    if "DSA" in selected_side:
        fold_side_features.append(side_drug_sim)
        fold_side_feature_names.append("DSA")
    return fold_drug_features, fold_side_features, fold_drug_feature_names, fold_side_feature_names

# ============================================================================
# Data Preprocessing Functions
# ============================================================================

def Extract_positive_negative_samples(DAL):
    """提取全部正样本与等量随机负样本，用于外层分层五折。"""
    # Flatten matrix to sample list [drug_idx, adr_idx, label]
    n_samples = DAL.shape[0] * DAL.shape[1] #num_drug*num_adr
    # 第三列保留原始 rating，不能强转 int，避免未来频率/强度标签被截断。
    interaction_target = np.zeros((n_samples, 3), dtype=np.float32)#[num_drug*num_adr, 3]
    
    k = 0
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k] = [i, j, DAL[i, j]]#[drug_idx, adr_idx, label]
            k += 1
    
    # Sort by label (negatives first, then positives)
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]#[num_drug*num_adr, 3] label为0排前面
    number_positive = np.count_nonzero(data_shuffle[:, 2])
    number_negative = n_samples - number_positive
    
    # Split positive and negative samples
    final_positive_sample = data_shuffle[number_negative:]
    negative_sample = data_shuffle[:number_negative]
    
    # 保持与旧实现一致：先生成完整随机排列，再取前面的等量负样本。
    sampled_indices = random.sample(range(number_negative), number_negative)
    final_negative_sample = negative_sample[sampled_indices[:number_positive]]
    return final_positive_sample, final_negative_sample


def compute_logit_adjustment_bias(args):
    """按RPU负采样比例做类别先验校正，让0.5阈值重新对应评估集的近似1:1先验。"""
    if args is None or not getattr(args, "use_logit_adjustment", False):
        return 0.0
    if not getattr(args, "use_rpu", False):
        return 0.0
    ratio = 10.0
    return float(np.log(ratio))


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_test(
        drug_feature, side_feature, data_train, data_val, data_test, fold, args, output_dir,
        raw_drug_token_features=None, raw_side_token_features=None,
        token_feature_mode="none"):
    """一折的训练和评估函数。
    
    Args:
        drug_feature: 药物相似性矩阵列表
        side_feature: 副作用相似性矩阵列表
        data_train: 训练样本
        data_val: 验证样本
        data_test: 最终测试样本
        fold: 当前折数
        args: 命令行参数
        output_dir: 当前训练输出目录
        
    Returns:
        Evaluation metrics (AUC, AUPR, RMSE, MAE, ACC, MCC)
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold} Training")
    print(f"{'='*60}\n")
    

    
    '''构建全局特征矩阵'''
    drug_entity_count = drug_feature[0].shape[0] if drug_feature else token_feature_entity_count(raw_drug_token_features)
    side_entity_count = side_feature[0].shape[0] if side_feature else token_feature_entity_count(raw_side_token_features)
    drug_features_matrix_global = build_dense_feature_matrix(drug_feature, drug_entity_count)
    side_features_matrix_global = build_dense_feature_matrix(side_feature, side_entity_count)
    
    global_drug_features_tensor = torch.FloatTensor(drug_features_matrix_global)
    global_side_features_tensor = torch.FloatTensor(side_features_matrix_global)
    print(f'全局特征矩阵: 药物 {global_drug_features_tensor.shape}, 副作用 {global_side_features_tensor.shape}')
    
    # 直接处理训练测试数据，无需额外函数
    data_train = np.array(data_train)
    data_val = np.array(data_val)
    data_test = np.array(data_test)
    
    if data_train.shape[1] == 3:
        data_train = to_weighted_train_samples(data_train)

    train_indices = (
        data_train[:, 0].astype(int),      # drug_indices
        data_train[:, 1].astype(int),      # side_indices
        data_train[:, 2].astype(np.float32),  # soft_label
        data_train[:, 3].astype(np.float32),  # rating
        data_train[:, 4].astype(np.float32),  # sample_weight
    )
    train_risk_components = data_train[:, 5:].astype(np.float32)
    
    val_indices = (
        data_val[:, 0].astype(int),
        data_val[:, 1].astype(int),
        data_val[:, 2]
    )

    test_indices = (
        data_test[:, 0].astype(int),
        data_test[:, 1].astype(int),
        data_test[:, 2]
    )
    
    '''构建训练集、验证集和最终测试集'''
    train_tensors = [
        torch.LongTensor(train_indices[0]),  # drug_indices
        torch.LongTensor(train_indices[1]),  # side_indices
        torch.FloatTensor(train_indices[2]),  # soft_label
        torch.FloatTensor(train_indices[3]),  # rating
        torch.FloatTensor(train_indices[4])   # sample_weight
    ]
    if train_risk_components.shape[1] > 0:
        train_tensors.append(torch.FloatTensor(train_risk_components))
    trainset = torch.utils.data.TensorDataset(*train_tensors)
    valset = torch.utils.data.TensorDataset(
        torch.LongTensor(val_indices[0]),
        torch.LongTensor(val_indices[1]),
        torch.FloatTensor(val_indices[2])
    )
    testset = torch.utils.data.TensorDataset(
        torch.LongTensor(test_indices[0]),
        torch.LongTensor(test_indices[1]),
        torch.FloatTensor(test_indices[2])
    )
    
    '''配置cuda加速'''
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # 全局特征矩阵很小，直接每个 fold 只搬一次到 GPU，避免每个 batch 重复整表拷贝。
    global_drug_features_tensor = global_drug_features_tensor.to(device, non_blocking=use_cuda)
    global_side_features_tensor = global_side_features_tensor.to(device, non_blocking=use_cuda)
    global_raw_drug_token_features = move_token_features_to_device(raw_drug_token_features, device)
    global_raw_side_token_features = move_token_features_to_device(raw_side_token_features, device)

    # 当前数据集已经在内存里，额外 worker 反而会放大 CPU 调度和拷贝开销。
    _val = torch.utils.data.DataLoader(
        valset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    _test = torch.utils.data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )
    print(f"[Fold {fold}] samples: train={len(trainset)}, val={len(valset)}, test={len(testset)}")

    _train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )

    # ChemProp 编码器已移除
    
    '''构建模型'''
    drug_feature_dims = [feature.shape[1] for feature in drug_feature]
    side_feature_dims = [feature.shape[1] for feature in side_feature]
    model = DGAPred(
        drugs_dim=sum(drug_feature_dims),
        sides_dim=sum(side_feature_dims),
        embed_dim=args.embed_dim,
        batchsize=args.batch_size,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        drug_feature_dims=drug_feature_dims,
        side_feature_dims=side_feature_dims,
        drug_token_vocab_sizes=[] if not raw_drug_token_features else [
            feature["vocab_size"] for feature in raw_drug_token_features.values()
        ],
        side_token_vocab_sizes=[] if not raw_side_token_features else [
            feature["vocab_size"] for feature in raw_side_token_features.values()
        ],
        token_feature_mode=token_feature_mode,
    ).to(device)
    if args.rpu_weight_mode == "learnable":
        component_count = train_risk_components.shape[1]
        if component_count == 0:
            raise ValueError("learnable 权重模式没有收到各特征风险分量。")
        # 初始 softmax 为等权融合，训练后只学习少量全局系数，不开放逐样本自由权重。
        model.rpu_fusion_logits = nn.Parameter(torch.zeros(component_count, device=device))
        print(f"[RPU-Weight] learnable fusion components={component_count}")
    
    '''构建损失函数和优化器'''
    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.BCEWithLogitsLoss()
    # 服务器环境里 foreach/multi_tensor Adam 偶发会在 step 阶段长时间卡住。
    # 这里关闭 foreach，换成更稳的逐参数更新实现，避免训练进程假死。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False
    )
    
    '''构建学习率调度器'''
    scheduler = None
    if args.use_scheduler:
        # 验证集综合分数连续不提升时降低学习率，让模型后期不要继续大步覆盖较好的排序边界。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=args.scheduler_patience,
            threshold=args.early_stop_delta,
            min_lr=args.min_lr
        )
    
    '''初始化训练指数变量'''
    AUC_mn = 0
    AUPR_mn = 0
    best_score = -np.inf

    rms_mn = 100000
    mae_mn = 100000
    endure_count = 0
    best_model_state = None  # Save best model state

    start = time.time()
    train_epoches = []
    test_epoches = []
    
    '''训练'''
    for epoch in range(1, args.epochs + 1):
        iter_loss_sum, step = train(
            model,
            _train_loader,
            optimizer,
            Regression_criterion,
            device,
            global_drug_features_tensor,
            global_side_features_tensor,
            global_raw_drug_token_features,
            global_raw_side_token_features,
            args=args,
        )  # 一个iterater
        train_epoch = iter_loss_sum/step
        train_epoches.append(train_epoch)
        
        v_i_auc, v_iPR_auc, v_rmse, v_mae, v_acc, v_mcc, v_ground_i, v_ground_u, v_ground_truth, v_pred1, v_pred2, val_iter_loss, val_step = test(model,
                                                                                                           _val,
                                                                                                           device,
                                                                                                           global_drug_features_tensor,
                                                                                                           global_side_features_tensor,
                                                                                                           global_raw_drug_token_features,
                                                                                                           global_raw_side_token_features,
                                                                                                           lossfunction1=Classification_criterion,
                                                                                                           lossfunction2=Regression_criterion,
                                                                                                           args=args)
                                                                                         
        test_epoch = val_iter_loss/val_step
        test_epoches.append(test_epoch)


        # AUC和AUPR都反映分类排序效果，用同一个综合分数保存最佳模型，避免只提升一个指标就覆盖模型。
        val_score = 0.5 * (v_i_auc + v_iPR_auc)
        is_better = val_score > best_score + args.early_stop_delta
        if is_better:
            best_score = val_score
            AUC_mn = v_i_auc
            AUPR_mn = v_iPR_auc
            rms_mn = v_rmse
            mae_mn = v_mae
            endure_count = 0
            # Save best model state
            best_model_state = deepcopy(model.state_dict())
        else:
            endure_count += 1

        if scheduler is not None:
            scheduler.step(val_score)

        current_lr = optimizer.param_groups[0]["lr"]
        print("Epoch: %d <Val after train-epoch> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f, LR: %.6g " % (
        epoch, v_rmse, v_mae, v_i_auc, v_iPR_auc, v_acc, v_mcc, current_lr))
        start = time.time()

        if endure_count >= args.early_stop_patience :
            break
    
    '''加载验证阶段表现最好的模型，再做最终测试'''
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Info] Loaded best model from validation (AUC: {AUC_mn:.5f}, AUPR: {AUPR_mn:.5f})")
    if args.rpu_weight_mode == "learnable":
        fusion_weights = torch.softmax(model.rpu_fusion_logits.detach(), dim=0).cpu().numpy()
        print(f"[RPU-Weight] best fusion weights={np.round(fusion_weights, 4).tolist()}")
    final_start = time.time()
    i_auc, iPR_auc, rmse, mae, acc, mcc, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step_ = test(
        model, _test, device, global_drug_features_tensor, global_side_features_tensor,
        global_raw_drug_token_features, global_raw_side_token_features,
        lossfunction1=Classification_criterion,
        lossfunction2=Regression_criterion,
        args=args
    )
    time_cost = time.time() - final_start
    print("Time: %.2f <Test> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f " % (
        time_cost, rmse, mae, i_auc, iPR_auc, acc, mcc))
    print('The best AUC/AUPR: %.5f / %.5f' % (i_auc, iPR_auc))
    print('The best ACC/MCC: %.5f / %.5f' % (acc, mcc))

    '''保存最终输出模型以及测试结果数据'''
    with open(os.path.join(output_dir,f'results.txt'),'a+') as f:
        # 只在第一折时保存超参数设置
        if fold == 1:
            f.write("\n===== Hyperparameters =====\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("===========================\n\n")
        f.write("Fold %d: AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f\n" % (fold, i_auc, iPR_auc, acc, mcc))
    with open(os.path.join(output_dir, f'model_fold{str(fold)}.pkl'), 'wb') as f:
        pickle.dump(model.state_dict(), f)
    print("Model saved to: %s" % os.path.join(output_dir, f'model_fold{str(fold)}.pkl'))
    with open(os.path.join(output_dir,f'testdata_fold{str(fold)}.pkl'),'wb') as f:
        test_data={"ground_truth":ground_truth,"pred_value":pred1}
        pickle.dump(test_data, f)
    print("Test data saved to: %s" % os.path.join(output_dir, f'testdata_fold{str(fold)}.pkl'))

    # plt.switch_backend('Agg')
    # fig = plt.figure()
    # pic1 = fig.add_subplot(2,1,1)
    # pic2 = fig.add_subplot(2,1,2)
    # pic1.plot(train_epoches,"skyblue",label="train_loss")
    # pic2.plot(test_epoches,"pink",label="test_loss")
    # pic1.legend()
    # pic2.legend()
    # pic1.set_ylabel("loss")
    # pic2.set_xlabel("epoch")
    # pic2.set_ylabel("loss")
    # plt.savefig(os.path.join(args.rawpath,"loss_curve.jpg"))

    return i_auc, iPR_auc, rmse, mae, acc, mcc

def train(model, train_loader, optimizer, lossfunction2, device,
          global_drug_features, global_side_features, global_drug_token_features=None,
          global_side_token_features=None, args=None):
    """训练函数 - 带进度条和实时指标"""
    model.train()
    
    avg_loss = 0.0
    losses = []  # 记录每个batch的loss

    # 创建进度条
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc="Training")
    for step, batch in pbar:
        drug_idx, side_idx, soft_labels, ratings, sample_weights = batch[:5]
        risk_components = batch[5] if len(batch) > 5 else None
        # E3 训练样本为五列：soft_label 负责分类，rating 只给真实正样本回归，sample_weight 表示负标签可信度。
        soft_labels = soft_labels.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        if risk_components is not None:
            risk_components = risk_components.to(device, non_blocking=True)
        real_positive_mask = ratings > 0
        
        optimizer.zero_grad()
        
        # 前向传播
        model_output = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            global_drug_token_features=global_drug_token_features,
            global_side_token_features=global_side_token_features,
        )
        # 模型内部使用 squeeze；这里拉平成一维，避免 batch_size=1 时退化成标量。
        logits, reconstruction = model_output
        logits = logits.view(-1)
        reconstruction = reconstruction.view(-1)
        
        # 标签平滑
        if args.label_smooth > 0:
            eps = float(args.label_smooth)
            y_target = (1.0 - eps) * soft_labels + 0.5 * eps
        else:
            y_target = soft_labels

        # 分类分支使用固定权重或可学习的多源风险融合。
        raw_bce = nn.functional.binary_cross_entropy_with_logits(logits, y_target, reduction='none')
        negative_mask = (soft_labels <= 0) & (ratings <= 0)
        dynamic_weights = sample_weights
        fusion_regularization = logits.new_zeros(())
        if args.rpu_weight_mode == "learnable":
            fusion_weights = torch.softmax(model.rpu_fusion_logits, dim=0)
            fused_risk = (risk_components * fusion_weights).sum(dim=1)
            learned_weights = torch.clamp(1.0 - fused_risk, min=MIN_NEG_WEIGHT, max=1.0)
            dynamic_weights = torch.where(negative_mask, learned_weights, sample_weights)
            # 弱正则避免训练初期系数直接集中到单一特征，同时保留自适应空间。
            uniform = torch.full_like(fusion_weights, 1.0 / len(fusion_weights))
            fusion_regularization = FUSION_REGULARIZATION * torch.mean((fusion_weights - uniform) ** 2)
        loss1 = (raw_bce * dynamic_weights).mean() + fusion_regularization

        # 回归分支只训练真实正样本，避免未观察负样本的 0 rating 污染强度预测。
        if real_positive_mask.any():
            loss2 = lossfunction2(reconstruction[real_positive_mask], ratings[real_positive_mask])
        else:
            loss2 = reconstruction.new_zeros(())
        total_loss = args.lambda_cls * loss1 + args.lambda_reg * loss2
        
        total_loss.backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        optimizer.step()
        
        # 更新指标
        batch_loss = total_loss.item()
        avg_loss += batch_loss
        losses.append(batch_loss)
        
        # 更新进度条显示（显示最近10个batch的平均loss）
        recent_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        pbar.set_postfix({'loss': f'{recent_loss:.4f}', 'avg': f'{avg_loss/(step+1):.4f}'})

    return avg_loss, step + 1

def test(model, test_loader, device, global_drug_features, global_side_features,
         global_drug_token_features=None, global_side_token_features=None,
         lossfunction1=None, lossfunction2=None, args=None):
    """测试函数 - 带进度条和实时指标"""
    model.eval()
    logit_adjustment_bias = compute_logit_adjustment_bias(args)
    
    pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    ground_u = []
    ground_i = []
    test_avg_loss = 0.0
    
    # 创建进度条，使用no_grad避免构建计算图
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    with torch.inference_mode():
      for step, (drug_idx, side_idx, ratings) in pbar:
        # 构建二分类标签
        ratings_cpu = ratings
        labels_cpu = (ratings_cpu > 0).float()
        ratings = ratings_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        
        ground_i.append(drug_idx.tolist())
        ground_u.append(side_idx.tolist())
        
        # 前向传播score_one:classfication score_two:regression
        scores_one, scores_two = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            global_drug_token_features=global_drug_token_features,
            global_side_token_features=global_side_token_features,
        )
        # 测试集最后一个 batch 可能只有1条样本，统一成一维张量后再计算指标。
        scores_one = scores_one.view(-1)
        scores_two = scores_two.view(-1)
        adjusted_scores_one = scores_one + logit_adjustment_bias
        positive_mask = labels > 0
        
        # 计算损失
        loss1 = lossfunction1(adjusted_scores_one, labels)#BCEWithLogitsLoss内部会做sigmoid
        # 验证/测试集按顺序取batch时更容易出现无正样本batch，必须跳过回归MSE，否则进度条会显示loss=nan。
        if positive_mask.any():
            loss2 = lossfunction2(scores_two[positive_mask], ratings[positive_mask])#在正样本上计算MSELoss
        else:
            loss2 = scores_two.new_zeros(())
        lambda_cls = args.lambda_cls if args is not None else 0.7
        lambda_reg = args.lambda_reg if args is not None else 0.2
        test_loss = lambda_cls * loss1 + lambda_reg * loss2
        test_avg_loss += test_loss.detach().item()
        
        # 收集预测结果。RPU使用大量未观察负样本时，logit先验校正只修正概率尺度，不改变排序关系。
        prob_one = torch.sigmoid(adjusted_scores_one)
        pred1.append(list(prob_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(ratings_cpu.tolist())
        label_truth.append(labels_cpu.tolist())
        
        # 更新进度条显示（显示平均loss）
        pbar.set_postfix({'loss': f'{test_avg_loss/(step+1):.4f}'})

    pred1 = np.array(sum(pred1, []), dtype = np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)



    iprecision, irecall, ithresholds = metrics.precision_recall_curve(label_truth,
                                                                      pred1,
                                                                      pos_label=1,
                                                                      sample_weight=None)
    iPR_auc = metrics.auc(irecall, iprecision)

    try:
        i_auc = metrics.roc_auc_score(label_truth, pred1)
    except ValueError:
        i_auc = 0

    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    # 依据0.5阈值计算二分类ACC与MCC
    y_pred_bin = (pred1 >= 0.5).astype(np.int32)
    acc = metrics.accuracy_score(label_truth, y_pred_bin)
    mcc = metrics.matthews_corrcoef(label_truth, y_pred_bin)

    return i_auc, iPR_auc, rmse, mae, acc, mcc, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step + 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 120,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 5e-4,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 5e-5,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default =128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--torch_threads', type=int, default=0,
                        metavar='N', help='PyTorch CPU算子线程数，0表示CUDA训练时自动限制到较低值')
    parser.add_argument('--torch_interop_threads', type=int, default=0,
                        metavar='N', help='PyTorch CPU算子调度线程数，0表示CUDA训练时自动限制到较低值')
    parser.add_argument('--rawpath', type=str, default='pythonPredict/DGAPred(Compare)/2drug-2side/DGAPred/data/',
                        metavar='STRING', help='rawpath')

    parser.add_argument('--similarity_path', type=str, default='pythonPredict/',
                        metavar='STRING', help='similarity matrices path')
    # 训练稳健性与正则化
    parser.add_argument('--dropout1', type=float, default=0.4,metavar='FLOAT', help='主特征编码阶段的dropout')
    parser.add_argument('--dropout2', type=float, default=0.2,metavar='FLOAT', help='Final prediction dropout rate')
    parser.add_argument('--label_smooth', type=float, default=0.0,metavar='FLOAT', help='二分类标签平滑系数，E3默认关闭')
    parser.add_argument('--grad_clip', type=float, default=0.5,metavar='FLOAT', help='梯度裁剪阈值，默认0表示关闭；开启示例：--grad_clip 0.5')
    parser.add_argument('--use_scheduler', action='store_true', help='启用基于验证AUC的ReduceLROnPlateau学习率调度',default=True)
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        metavar='N', help='验证综合分数连续多少轮不提升后降低学习率')
    parser.add_argument('--early_stop_patience', type=int, default=8,
                        metavar='N', help='验证综合分数连续多少轮不提升后提前停止训练')
    parser.add_argument('--early_stop_delta', type=float, default=1e-4,
                        metavar='FLOAT', help='验证综合分数至少提升多少才算真正变好')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        metavar='FLOAT', help='学习率调度器允许降到的最小学习率')
    parser.add_argument('--drug_features', type=str, default='DGen,CS,DSA',
                        help='药物相似度特征，允许 DGen、CS、DSA')
    parser.add_argument('--adr_features', type=str, default='MESH,GDA,DSA',
                        help='ADR相似度特征，允许 MESH、GDA、DSA')
    parser.add_argument('--raw_feature_mode', choices=['none', 'replace'], default='none',
                        help='原始特征使用方式：replace为预测模型仅使用原始特征')
    parser.add_argument('--raw_drug_features', type=str, default='DGen,CS',
                        help='药物原始特征，允许 DGen、CS')
    parser.add_argument('--raw_adr_features', type=str, default='MESH,GDA',
                        help='ADR原始特征，允许 MESH、GDA')
    parser.add_argument('--val_source', choices=['train', 'test'], default='train',
                        help='验证集来源：train为严格口径，test仅用于复现旧口径')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        metavar='FLOAT', help='验证集划分比例')
    parser.add_argument('--lambda_cls', type=float, default=0.7,
                        metavar='FLOAT', help='分类损失权重')
    parser.add_argument('--lambda_reg', type=float, default=0.2,
                        metavar='FLOAT', help='回归损失权重')
    parser.add_argument('--use_logit_adjustment',default=True, action='store_true',
                        help='启用RPU类别先验logit校正，修正负采样导致的概率整体偏低')
    parser.add_argument('--use_rpu',default=True, action='store_true',
                        help='启用RPU-DGAPred E3训练样本构造')
    parser.add_argument('--rpu_weight_mode', choices=['none', 'learnable'], default='none',
                        help='RPU负样本权重模式：none固定，ema预测风险滑动更新，learnable学习特征融合系数')
    parser.add_argument('--rpu_use_consensus_pseudo',default=True, action='store_true',
                        help='启用多视图共识伪阳性：多个特征视图一致高分的未知pair会以弱标签加入训练')
    args = parser.parse_args()
    drug_tokens = parse_feature_tokens(args.drug_features)
    adr_tokens = parse_feature_tokens(args.adr_features)
    raw_drug_tokens = parse_feature_tokens(args.raw_drug_features)
    raw_adr_tokens = parse_feature_tokens(args.raw_adr_features)
    if not drug_tokens <= {"DGEN", "CS", "DSA"}:
        raise ValueError("drug_features 只允许 DGen、CS、DSA。")
    if not adr_tokens <= {"MESH", "GDA", "DSA"}:
        raise ValueError("adr_features 只允许 MESH、GDA、DSA。")
    if not raw_drug_tokens <= {"DGEN", "CS"}:
        raise ValueError("raw_drug_features 只允许 DGen、CS。")
    if not raw_adr_tokens <= {"MESH", "GDA"}:
        raise ValueError("raw_adr_features 只允许 MESH、GDA。")
    if args.rpu_weight_mode != "none" and not args.use_rpu:
        raise ValueError("learnable 权重模式只能在启用 RPU 时使用。")
    normalize_raw_configuration(args)
    print(
        f"[RawFeatureConfig] mode={args.raw_feature_mode}, "
        f"drug={args.raw_drug_features or 'none'}, adr={args.raw_adr_features or 'none'}"
    )
    print(f"[RPU-Weight] mode={args.rpu_weight_mode}")
    configure_cpu_threads(
        args.torch_threads,
        args.torch_interop_threads,
        prefer_cuda=torch.cuda.is_available()
    )
    if args.use_logit_adjustment:
        print(f"[LogitAdjustment] bias={compute_logit_adjustment_bias(args):.4f}")
    args.rawpath, args.similarity_path = ensure_training_feature_cache(
        args.rawpath,
        args.similarity_path
    )
    
    drug_ids, adr_ids, drug_side = load_label(args)
    print("drug_side shape:",pd.DataFrame(drug_side).shape)

    # 加载药物和不良反应特征；读取时再次校验顺序，防止特征矩阵与标签矩阵错位。
    drug_feature, drug_feature_names = load_drug_feature(drug_ids, args)
    side_feature, side_feature_names = load_adr_feature(adr_ids, args)
    raw_drug_dense_features = {}
    raw_drug_token_features = {}
    raw_side_dense_features = {}
    raw_side_token_features = {}
    if args.raw_feature_mode != 'none':
        raw_drug_dense_features, raw_drug_token_features = load_raw_drug_features(drug_ids, args)
        raw_side_dense_features, raw_side_token_features = load_raw_adr_features(adr_ids, args)
        if not (raw_drug_dense_features or raw_drug_token_features or raw_side_dense_features or raw_side_token_features):
            raise ValueError('启用原始特征模式时，至少需要指定一个原始特征。')
        print(
            f"[RawFeature] mode={args.raw_feature_mode}, "
            f"drug_dense={list(raw_drug_dense_features)}, drug_token={list(raw_drug_token_features)}, "
            f"adr_dense={list(raw_side_dense_features)}, adr_token={list(raw_side_token_features)}"
        )
    
    # 外层交叉验证仍使用1:1正负样本，保证验证/测试口径和原DGAPred对照一致。
    final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values)
    final_sample = np.vstack((final_positive_sample, final_negative_sample))
    X = final_sample[:, 0::]
    final_target = final_sample[:, final_sample.shape[1] - 1]
    y = final_target
    data = []
    data_x = []
    data_y = []
    for i in range(X.shape[0]):
        data_x.append((X[i, 0], X[i, 1]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 0], X[i, 1], X[i, 2]))
    
    # 正常五折训练
    fold = 1
    kfold = StratifiedKFold(5, random_state=5, shuffle=True)
    total_auc, total_pr_auc, total_rmse, total_mae = [], [], [], []
    total_acc, total_mcc = [], []
    #建立输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    normalized_rawpath = os.path.normpath(args.rawpath)#规范化路径，解决路径中的冗余和不一致
    output_dir = os.path.join(normalized_rawpath, f'output_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    #开始五折交叉验证
    for k, (train_split, test_split) in enumerate(kfold.split(data_x, data_y)):
        print("==================================fold {} start".format(fold))
        data = np.array(data)
        fold_train_data = data[train_split].tolist()
        test_data = data[test_split].tolist()
        if args.val_source == 'train':
            fold_train_data, val_data = split_train_val(
                fold_train_data,
                val_ratio=args.val_ratio,
                seed=42
            )
        else:
            val_data, test_data = split_val_test(
                test_data,
                val_ratio=args.val_ratio,
                seed=42
            )
        print(
            f"[Fold {fold}] val_source={args.val_source}, "
            f"train={len(fold_train_data)}, val={len(val_data)}, test={len(test_data)}",
            flush=True
        )
        risk_drug_feature, risk_side_feature, risk_drug_feature_names, risk_side_feature_names = add_dsa_features(
            drug_feature,
            side_feature,
            drug_feature_names,
            side_feature_names,
            drug_side,
            val_data + test_data,
            args
        )
        model_drug_feature = risk_drug_feature
        model_side_feature = risk_side_feature
        model_drug_token_features = None
        model_side_token_features = None
        token_feature_mode = "none"
        if args.raw_feature_mode == 'replace':
            model_drug_feature, model_drug_feature_names = raw_dense_feature_list(raw_drug_dense_features)
            model_side_feature, model_side_feature_names = raw_dense_feature_list(raw_side_dense_features)
            model_drug_feature, model_side_feature, model_drug_feature_names, model_side_feature_names = add_raw_dsa_features(
                model_drug_feature,
                model_side_feature,
                model_drug_feature_names,
                model_side_feature_names,
                drug_side,
                val_data + test_data,
                args,
            )
            model_drug_token_features = raw_drug_token_features
            model_side_token_features = raw_side_token_features
            token_feature_mode = "replace"
            print(f"[RawFeature] Fold {fold} prediction uses raw-only views: "
                  f"drug={model_drug_feature_names + list(raw_drug_token_features)}, "
                  f"adr={model_side_feature_names + list(raw_side_token_features)}")
        if args.use_rpu:
            fold_train_data = build_rpu_train_samples(
                fold_train_data,
                val_data + test_data,
                drug_side.values,
                risk_drug_feature,
                risk_side_feature,
                args,
                fold,
                risk_drug_feature_names,
                risk_side_feature_names,
            )
        else:
            fold_train_data = to_weighted_train_samples(fold_train_data)
        auc, PR_auc, rmse, mae, acc, mcc = train_test(
            model_drug_feature,
            model_side_feature,
            fold_train_data,
            val_data,
            test_data,
            fold,
            args,
            output_dir,
            raw_drug_token_features=model_drug_token_features,
            raw_side_token_features=model_side_token_features,
            token_feature_mode=token_feature_mode,
        )
        total_rmse.append(rmse)
        total_mae.append(mae)
        total_auc.append(auc)
        total_pr_auc.append(PR_auc)
        total_acc.append(acc)
        total_mcc.append(mcc)
        print("==================================fold {} end".format(fold))#每个Fold之后输出指标值平均值
        fold += 1
        print('Total_AUC:')
        print(np.mean(total_auc))
        print('Total_AUPR:')
        print(np.mean(total_pr_auc))
        print('Total_RMSE:')
        print(np.mean(total_rmse))
        print('Total_MAE:')
        print(np.mean(total_mae))
        print('Total_ACC:')
        print(np.mean(total_acc))
        print('Total_MCC:')
        print(np.mean(total_mcc))
        sys.stdout.flush()
