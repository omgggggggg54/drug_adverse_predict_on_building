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

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from model.model import DGAPred
from utils.data_utils import jaccard_similarity
from utils.feature_generation import ensure_training_feature_cache, read_ordered_square_feature
from utils.gene_feature_v2 import ensure_gene_tfidf_svd
from utils.raw_feature_generation import (
    ensure_adr_mesh_raw,
    ensure_drug_fingerprint_raw,
    read_ordered_raw_feature,
    read_ordered_mesh_token_feature,
)
from utils.rpu_utils import build_rpu_train_samples, to_weighted_train_samples
from utils.topology_score import (
    MULTISCALE_FEATURE_NAMES,
    assert_hidden_pairs_are_masked,
    build_topology_scorer,
)
from utils.topology_calibration import (
    classification_metrics,
    fit_topology_calibration,
    predict_topology_calibration,
)

# 设置随机种子确保可复现性
SEED = 42
# 外层五折沿用历史 random_state=5；其余随机入口统一使用 SEED。
OUTER_FOLD_SEED = 5
OUTER_FOLD_COUNT = 5


def set_random_seed(seed):
    """统一设置 Python、NumPy 和 PyTorch 的随机状态。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_random_seed(SEED)
# 固定 cuDNN/CUDA 算法，保证相同 seed 的重复运行不会因算子选择产生额外波动。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    """加载训练标签矩阵。"""
    cache_path = os.path.join(args.similarity_path, "drug_side.csv")
    print(f"[Cache] Loading from cache: {cache_path}")
    drug_side = pd.read_csv(cache_path, header=0, index_col=0)
    return list(drug_side.index), list(drug_side.columns), drug_side


SIMILARITY_DRUG_FEATURES = [
    ("DGen", "drug_DGen_sim.csv"),
    ("CS", "drug_rdkit.csv"),
]

SIMILARITY_ADR_FEATURES = [
    ("MESH", "side_mesh_sim.csv"),
    ("GDA", "adr_GDisease_sim.csv"),
]


def load_similarity_features(drug_ids, adr_ids, args):
    """加载固定相似度视图，供 RPU 及 similarity 模式预测共同使用。"""
    drug_features = []
    side_features = []
    drug_feature_names = []
    side_feature_names = []

    for name, filename in SIMILARITY_DRUG_FEATURES:
        feature = read_ordered_square_feature(
            os.path.join(args.similarity_path, filename), drug_ids, f"药物相似度 {name}"
        )
        drug_features.append(feature)
        drug_feature_names.append(name)

    for name, filename in SIMILARITY_ADR_FEATURES:
        feature = read_ordered_square_feature(
            os.path.join(args.similarity_path, filename), adr_ids, f"ADR 相似度 {name}"
        )
        side_features.append(feature)
        side_feature_names.append(name)

    print(f"[Similarity] drug={drug_feature_names}, adr={side_feature_names}")
    return drug_features, side_features, drug_feature_names, side_feature_names


def split_train_val(data_train, val_ratio=0.2):
    """未启用拓扑校准时，从外层训练折中固定切出早停验证集。"""
    # 此路径不需要独立校准集：验证集只负责 CNN 早停，最终测试集仍完全隔离。
    # 输入每行是 [drug_idx, adr_idx, rating]；返回的两组样本保持同一列格式。
    samples = np.asarray(data_train)
    labels = (samples[:, 2] > 0).astype(np.int32)
    n_splits = max(2, int(round(1.0 / val_ratio)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(splitter.split(samples[:, :2], labels))
    return samples[train_idx].tolist(), samples[val_idx].tolist()


def split_train_val_calibration(data_train, val_ratio=0.1, calibration_ratio=0.1):
    """按正负标签固定切分基础训练、验证与校准集。"""
    # 只保留最必要的标签分层，不再引入药物度数、ADR 度数和稀有组合合并。
    # 第一阶段从外层训练折取出校准集，第二阶段再从剩余样本取出验证集。
    if val_ratio <= 0 or calibration_ratio <= 0 or val_ratio + calibration_ratio >= 1:
        raise ValueError("早停验证和校准比例必须大于 0，且总和小于 1。")

    samples = np.asarray(data_train)
    # 每行格式为 [drug_idx, adr_idx, rating]；分层只使用 rating 是否大于零。
    labels = (samples[:, 2] > 0).astype(np.int32)
    calibration_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=calibration_ratio,
        random_state=SEED,
    )
    remaining_idx, calibration_idx = next(
        calibration_splitter.split(samples[:, :2], labels)
    )
    remaining = samples[remaining_idx]
    remaining_labels = labels[remaining_idx]

    # 校准集已经取走 calibration_ratio，因此验证比例需换算到剩余样本中。
    # 默认值下为 0.1 / 0.9，最终得到约 80%/10%/10%。
    validation_ratio_in_remaining = val_ratio / (1.0 - calibration_ratio)
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_ratio_in_remaining,
        random_state=SEED,
    )
    base_idx, validation_idx = next(
        validation_splitter.split(remaining[:, :2], remaining_labels)
    )
    return (
        remaining[base_idx].tolist(),        # 基础训练集：训练 CNN 并构造拓扑矩阵 A。
        remaining[validation_idx].tolist(),  # 验证集：只用于 CNN 早停。
        samples[calibration_idx].tolist(),   # 校准集：只用于拟合线性排序校准器。
    )


def assert_disjoint_pair_splits(*datasets):
    """校验一个外层折内的基础训练、验证、校准和测试 pair 互不重叠。"""
    pair_sets = []
    for data in datasets:
        samples = np.asarray(data)
        pair_sets.append({(int(row[0]), int(row[1])) for row in samples})
    for index, current in enumerate(pair_sets):
        for other in pair_sets[index + 1:]:
            if current & other:
                raise RuntimeError("训练、验证、校准和测试 pair 存在重叠，终止以避免泄露。")


def load_tfidf_svd_features(drug_ids, adr_ids, args, gene_svd):
    """加载 TF-IDF-SVD dense 特征和 ADR MESH token 特征。"""
    drug_dense_features = {}
    side_dense_features = {}
    output_path = os.path.join(args.similarity_path, "drug_rdkit_raw.npz")
    ensure_drug_fingerprint_raw(args.similarity_path, drug_ids, "drug_rdkit_raw.npz", "rdkit")
    drug_dense_features["CS"] = read_ordered_raw_feature(output_path, drug_ids, "CS")

    mesh_path = os.path.join(args.similarity_path, "adr_mesh_raw_tokens.npz")
    ensure_adr_mesh_raw(args.rawpath, args.similarity_path, adr_ids, "adr_mesh_raw_tokens.npz")#编码成token id存成npz文件
    mesh_token_feature = read_ordered_mesh_token_feature(mesh_path, adr_ids)
    drug_dense_features["DGEN_TFIDF_SVD"] = gene_svd["drug_svd"]
    side_dense_features["GDA_TFIDF_SVD"] = gene_svd["adr_svd"]

    print(
        f"[RawFeature] drug_dense={list(drug_dense_features)}, "
        f"adr_dense={list(side_dense_features)}, adr_token=MESH"#输出keys
    )
    return drug_dense_features, side_dense_features, mesh_token_feature


def build_tfidf_svd_prediction_features(
        drug_dense_features, side_dense_features, drug_side, hidden_data):
    """构造 TF-IDF-SVD 模式预测输入，并追加当前 fold 可见的二值 DSARaw。"""
    visible = drug_side.values.astype(np.float32, copy=True)
    hidden = np.asarray(hidden_data)
    visible[hidden[:, 0].astype(int), hidden[:, 1].astype(int)] = 0.0
    visible = (visible > 0).astype(np.float32)

    drug_features = list(drug_dense_features.values()) + [visible]
    side_features = list(side_dense_features.values()) + [visible.T]
    drug_feature_names = list(drug_dense_features) + ["DSARaw"]
    side_feature_names = list(side_dense_features) + ["DSARaw"]
    return drug_features, side_features, drug_feature_names, side_feature_names


def build_dense_feature_matrix(features):
    """拼接实体的 dense 特征。"""
    return np.hstack(features).astype(np.float32)


def move_mesh_token_feature_to_device(mesh_token_feature, device):
    """将 ADR MESH token 缓存一次性转为训练设备上的只读张量。"""
    return (
        torch.as_tensor(mesh_token_feature["token_ids"], dtype=torch.long, device=device),
        torch.as_tensor(mesh_token_feature["offsets"], dtype=torch.long, device=device),
    )

def build_fold_similarity_features(
        drug_features, side_features, drug_feature_names, side_feature_names,
        drug_side, hidden_data):
    """追加 fold-safe Jaccard DSA，供固定 RPU 与 similarity 模式共同使用。"""
    drug_side_for_sim = drug_side.values.copy()
    hidden_array = np.array(hidden_data)
    drug_side_for_sim[hidden_array[:, 0].astype(int), hidden_array[:, 1].astype(int)] = 0
    drug_side_sim = jaccard_similarity(drug_side_for_sim)
    side_drug_sim = jaccard_similarity(drug_side_for_sim.T)

    print(f"[RPU-DSA] drug similarity: {drug_side_sim.shape}")
    print(f"[RPU-DSA] ADR similarity: {side_drug_sim.shape}")
    fold_drug_features = list(drug_features)
    fold_side_features = list(side_features)
    fold_drug_feature_names = list(drug_feature_names)
    fold_side_feature_names = list(side_feature_names)
    fold_drug_features.append(drug_side_sim)
    fold_drug_feature_names.append("DSA")
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
    
    # 固定局部随机生成器，避免其他模块消耗随机状态后改变外层测试样本。
    sampled_indices = random.Random(SEED).sample(range(number_negative), number_negative)
    final_negative_sample = negative_sample[sampled_indices[:number_positive]]
    return final_positive_sample, final_negative_sample


def compute_logit_adjustment_bias(use_rpu):
    """按固定 RPU 负采样比例校正分类阈值对应的先验。"""
    return float(np.log(10.0)) if use_rpu else 0.0


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_test(
        drug_feature, side_feature, data_train, data_val, data_test, fold, args, output_dir,
        mesh_token_feature, calibration_data=None, topology_scorer=None):
    """完成一个外层折的 CNN 训练、线性校准和最终评估。

    当启用拓扑校准时，外层训练折已经在主流程中划分为基础训练集、
    早停验证集和独立校准集。CNN 只使用基础训练集训练，并由验证集
    选择最佳权重；最佳 CNN 的原始分类 logit 与 16 维多尺度拓扑特征随后在
    校准集上拟合标准化线性 RankNet。校准器固定
    后，外层测试集只进行一次最终预测，测试标签不参与任何训练或选择。

    Args:
        drug_feature: 当前折使用的药物特征矩阵列表。
        side_feature: 当前折使用的副作用特征矩阵列表。
        data_train: CNN 基础训练样本；不包含验证、校准和外层测试 pair。
        data_val: CNN 早停验证样本；只用于选择最佳 CNN 权重。
        data_test: 外层最终测试样本；只在全部模型固定后进行一次评估。
        fold: 当前外层折编号。
        args: 命令行参数和训练配置。
        output_dir: 当前运行的结果输出目录。
        mesh_token_feature: ADR MESH token 特征；未使用时为 None。
        calibration_data: 独立校准样本；未启用拓扑校准时为空或 None。
        topology_scorer: 仅由基础训练正边构造的拓扑特征生成器。

    Returns:
        当前外层折的 AUC、AUPR、RMSE、MAE、ACC 和 MCC。
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold} Training")
    print(f"{'='*60}\n")
    

    
    '''构建全局特征矩阵'''
    drug_features_matrix_global = build_dense_feature_matrix(drug_feature)
    side_features_matrix_global = build_dense_feature_matrix(side_feature)
    
    global_drug_features_tensor = torch.FloatTensor(drug_features_matrix_global)
    global_side_features_tensor = torch.FloatTensor(side_features_matrix_global)
    print(f'全局特征矩阵: 药物 {global_drug_features_tensor.shape}, 副作用 {global_side_features_tensor.shape}')
    
    # 直接处理训练测试数据，无需额外函数
    data_train = np.array(data_train)
    data_val = np.array(data_val)
    data_test = np.array(data_test)
    
    train_indices = (
        data_train[:, 0].astype(int),      # drug_indices
        data_train[:, 1].astype(int),      # side_indices
        data_train[:, 2].astype(np.float32),  # soft_label
        data_train[:, 3].astype(np.float32),  # rating
        data_train[:, 4].astype(np.float32),  # sample_weight
    )
    
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    # 全局特征矩阵很小，直接每个 fold 只搬一次到 GPU，避免每个 batch 重复整表拷贝。
    global_drug_features_tensor = global_drug_features_tensor.to(device, non_blocking=use_cuda)
    global_side_features_tensor = global_side_features_tensor.to(device, non_blocking=use_cuda)
    global_mesh_token_feature = (
        move_mesh_token_feature_to_device(mesh_token_feature, device)
        if mesh_token_feature is not None else None
    )

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

    '''构建模型'''
    drug_feature_dims = [feature.shape[1] for feature in drug_feature]
    side_feature_dims = [feature.shape[1] for feature in side_feature]
    model = DGAPred(
        drugs_dim=sum(drug_feature_dims),
        sides_dim=sum(side_feature_dims),
        embed_dim=args.embed_dim,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        drug_feature_dims=drug_feature_dims,
        side_feature_dims=side_feature_dims,
        mesh_vocab_size=(
            mesh_token_feature["vocab_size"]
            if mesh_token_feature is not None else None
        ),
    ).to(device)
    
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

    endure_count = 0
    best_model_state = None  # Save best model state

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
            global_mesh_token_feature,
            args=args,
        )  # 一个iterater
        v_i_auc, v_iPR_auc, v_rmse, v_mae, v_acc, v_mcc, _, _, _, _ = test(model,
                                                                                                           _val,
                                                                                                           device,
                                                                                                           global_drug_features_tensor,
                                                                                                           global_side_features_tensor,
                                                                                                           global_mesh_token_feature,
                                                                                                           lossfunction1=Classification_criterion,
                                                                                                           lossfunction2=Regression_criterion,
                                                                                                           args=args,
                                                                                                           progress_name="Validation")
                                                                                         
        # AUC和AUPR都反映分类排序效果，用同一个综合分数保存最佳模型，避免只提升一个指标就覆盖模型。
        val_score = 0.5 * (v_i_auc + v_iPR_auc)
        is_better = val_score > best_score + args.early_stop_delta
        if is_better:
            best_score = val_score
            AUC_mn = v_i_auc
            AUPR_mn = v_iPR_auc
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
        if endure_count >= args.early_stop_patience :
            break
    
    '''加载验证阶段表现最好的模型，再做最终测试'''
    
    model.load_state_dict(best_model_state)
    print(f"\n[Info] Loaded best model from validation (AUC: {AUC_mn:.5f}, AUPR: {AUPR_mn:.5f})")
    topology_calibration = None
    topology_feature_names = []
    if args.use_topology_fusion:
        print("[TopologyFusion] 开始构建校准特征并拟合线性校准器")

        # 1. 特征构建：固定 CNN 后，独立校准集的 CNN logit 与折内拓扑特征拼成 17 维输入。
        # DataLoader 关闭 shuffle，使 pair、标签、logit 和拓扑特征始终按相同顺序拼接。
        # 校准集没有参与 CNN 反向传播，也没有参与最佳 epoch 的选择。
        calibration_array = np.asarray(calibration_data)
        calibration_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.LongTensor(calibration_array[:, 0].astype(int)),
                torch.LongTensor(calibration_array[:, 1].astype(int)),
                torch.FloatTensor(calibration_array[:, 2]),
            ),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
        calibration_pairs, calibration_labels, calibration_logits = collect_raw_predictions(
            model, calibration_loader, device, global_drug_features_tensor,
            global_side_features_tensor, global_mesh_token_feature, "Calibration logits"
        )
        # 第一列保留未经过 RPU 先验偏置修正的 CNN 原始 logit；其余 16 列来自基础训练正边。
        # 因而 calibration_features 形状为 (校准样本数, 17)，每一行只对应一个校准 pair。
        calibration_features = np.column_stack((
            calibration_logits,
            topology_scorer.pair_features(calibration_pairs[:, 0], calibration_pairs[:, 1]),
        ))

        # 2. 模型拟合：只在独立校准集上拟合标准化线性 BCE + RankNet 校准器。
        topology_calibration = fit_topology_calibration(calibration_features, calibration_labels)#拟合线性校准器
        # 该字典保存标准化均值、尺度、17 个线性权重和偏置；写入检查点后可独立重现实验推理。
        topology_feature_names = ["cnn_logit"] + MULTISCALE_FEATURE_NAMES
        print("[TopologyFusion] 标准化特征权重：")
        for name, weight in zip(topology_feature_names, topology_calibration["weight"].tolist()):
            print(f"  {name}: {weight:.6f}")

    # 校准器固定后，外层测试集只在这里进行一次最终推理。
    # 测试集既不参与 CNN 早停，也不参与 scaler、线性权重、RankNet 配对或阈值选择。
    final_start = time.time()
    i_auc, iPR_auc, rmse, mae, acc, mcc, ground_truth, pred1, raw_pred1, pred2 = test(
        model, _test, device, global_drug_features_tensor, global_side_features_tensor,
        global_mesh_token_feature,
        lossfunction1=Classification_criterion,
        lossfunction2=Regression_criterion,
        args=args,
        topology_calibration=topology_calibration,
        topology_scorer=topology_scorer,
        progress_name="Outer test",
    )
    time_cost = time.time() - final_start
    raw_labels = (ground_truth > 0).astype(np.float32)
    raw_auc, raw_aupr, _, _ = classification_metrics(raw_labels, raw_pred1)
    print(f"[CNN] 原始分类 AUC/AUPR: {raw_auc:.5f} / {raw_aupr:.5f}")
    print("Time: %.2f <Test> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f " % (
        time_cost, rmse, mae, i_auc, iPR_auc, acc, mcc))
    print('The best AUC/AUPR: %.5f / %.5f' % (i_auc, iPR_auc))
    print('The best ACC/MCC: %.5f / %.5f' % (acc, mcc))

    '''保存最终输出模型以及测试结果数据'''
    with open(os.path.join(output_dir, 'results.txt'), 'a+') as f:
        # 只在第一折时保存超参数设置
        if fold == 1:
            f.write("\n===== Hyperparameters =====\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("===========================\n\n")
        f.write(
            "Fold %d: CNN_AUC: %.5f, CNN_AUPR: %.5f, AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f\n" %
            (fold, raw_auc, raw_aupr, i_auc, iPR_auc, acc, mcc)
        )
    checkpoint_path = os.path.join(output_dir, f'model_fold{str(fold)}.pth')
    checkpoint = {"model_state_dict": model.state_dict()}
    if topology_calibration is not None:
        # 保存完整校准器而非只保存预测值，便于后续加载同一 CNN 后对新 pair 做一致推理。
        checkpoint.update({
            "topology_calibration": topology_calibration,
            "topology_feature_names": topology_feature_names,
        })
    torch.save(checkpoint, checkpoint_path)
    print("Model saved to: %s" % checkpoint_path)
    with open(os.path.join(output_dir,f'testdata_fold{str(fold)}.pkl'),'wb') as f:
        test_data={
            "ground_truth": ground_truth,
            # 保留校准前概率，供逐折比较 CNN 本身与拓扑校准带来的变化。
            "cnn_pred_value": raw_pred1,
            # pred_value 为启用校准后的概率；未启用拓扑校准时它与 cnn_pred_value 相同。
            "pred_value": pred1,
        }
        pickle.dump(test_data, f)
    print("Test data saved to: %s" % os.path.join(output_dir, f'testdata_fold{str(fold)}.pkl'))

    return i_auc, iPR_auc, rmse, mae, acc, mcc

def train(model, train_loader, optimizer, lossfunction2, device,
          global_drug_features, global_side_features, global_mesh_token_feature, args):
    """训练函数 - 带进度条和实时指标"""
    model.train()
    
    avg_loss = 0.0
    losses = []  # 记录每个batch的loss

    # 创建进度条
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc="Training")
    for step, batch in pbar:
        drug_idx, side_idx, soft_labels, ratings, sample_weights = batch[:5]
        # E3 训练样本为五列：soft_label 负责分类，rating 只给真实正样本回归，sample_weight 表示负标签可信度。
        soft_labels = soft_labels.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        real_positive_mask = ratings > 0
        
        optimizer.zero_grad()
        
        # 前向传播
        model_output = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            global_mesh_token_feature=global_mesh_token_feature,
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

        # 分类分支直接使用固定 RPU 样本权重。
        raw_bce = nn.functional.binary_cross_entropy_with_logits(logits, y_target, reduction='none')
        loss1 = (raw_bce * sample_weights).mean()

        # 回归分支只训练真实正样本，避免未观察负样本的 0 rating 污染强度预测。
        if real_positive_mask.any():
            loss2 = lossfunction2(reconstruction[real_positive_mask], ratings[real_positive_mask])
        else:
            loss2 = reconstruction.new_zeros(())
        total_loss = args.lambda_cls * loss1 + args.lambda_reg * loss2
        
        total_loss.backward()
        if args.grad_clip > 0:
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


def collect_raw_predictions(model, data_loader, device, global_drug_features,
                            global_side_features, global_mesh_token_feature, progress_name):
    """收集未做 RPU 先验修正的分类 logit，供独立校准集拟合融合器。"""
    # 这里返回的是 sigmoid 之前的 logit。线性校准器本身会重新学习偏置和特征权重，
    # 因而不应在此叠加只服务于原 CNN 概率解释的 RPU logit_adjustment_bias。
    model.eval()
    pairs = []
    labels = []
    logits_all = []
    with torch.inference_mode():
        for drug_idx, side_idx, ratings in tqdm(data_loader, desc=progress_name, leave=False):
            logits, _ = model(
                drug_indices=drug_idx,
                side_indices=side_idx,
                device=device,
                global_drug_features=global_drug_features,
                global_side_features=global_side_features,
                global_mesh_token_feature=global_mesh_token_feature,
            )
            pairs.append(np.column_stack((drug_idx.numpy(), side_idx.numpy())))
            labels.append((ratings.numpy() > 0).astype(np.float32))
            logits_all.append(logits.view(-1).detach().cpu().numpy())
    return (
        # 三个数组由同一批次循环顺序收集，行号一一对应。
        np.concatenate(pairs).astype(np.int64),
        np.concatenate(labels).astype(np.float32),
        np.concatenate(logits_all).astype(np.float32),
    )


def test(model, test_loader, device, global_drug_features, global_side_features,
         global_mesh_token_feature,
         lossfunction1, lossfunction2, args, topology_calibration=None, topology_scorer=None,
         progress_name="Testing"):
    """测试函数 - 带进度条和实时指标"""
    model.eval()
    logit_adjustment_bias = compute_logit_adjustment_bias(args.use_rpu)
    
    pred1 = []
    raw_pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    test_avg_loss = 0.0
    
    # 创建进度条，使用no_grad避免构建计算图
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=progress_name)
    with torch.inference_mode():
      for step, (drug_idx, side_idx, ratings) in pbar:
        # 构建二分类标签
        ratings_cpu = ratings
        labels_cpu = (ratings_cpu > 0).float()
        ratings = ratings_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        
        # 前向传播score_one:classfication score_two:regression
        scores_one, scores_two = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            global_mesh_token_feature=global_mesh_token_feature,
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
        test_loss = args.lambda_cls * loss1 + args.lambda_reg * loss2
        test_avg_loss += test_loss.detach().item()
        
        # 保留未校准 CNN 概率，便于每折直接比较校准前后的 AUC/AUPR。
        raw_probability = torch.sigmoid(adjusted_scores_one).cpu().numpy()
        # 未启用校准时保持原 CNN 概率；启用后由固定线性校准器替换分类概率。
        if topology_calibration is None:
            probability = raw_probability
        else:
            # 测试阶段只读取校准集已保存的参数；pair_features 也只读取基础训练折预计算的矩阵。
            # 因此这一分支没有任何基于当前测试标签的拟合或统计操作。
            features = np.column_stack((
                scores_one.detach().cpu().numpy(),
                topology_scorer.pair_features(drug_idx.numpy(), side_idx.numpy()),
            ))
            probability = predict_topology_calibration(topology_calibration, features)
        pred1.append(list(probability))
        raw_pred1.append(list(raw_probability))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(ratings_cpu.tolist())
        label_truth.append(labels_cpu.tolist())
        
        # 更新进度条显示（显示平均loss）
        pbar.set_postfix({'loss': f'{test_avg_loss/(step+1):.4f}'})

    pred1 = np.array(sum(pred1, []), dtype = np.float32)
    raw_pred1 = np.array(sum(raw_pred1, []), dtype=np.float32)
    pred2 = np.array(sum(pred2, []), dtype=np.float32)

    ground_truth = np.array(sum(ground_truth, []), dtype = np.float32)
    label_truth = np.array(sum(label_truth, []), dtype=np.float32)

    one_label_index = np.nonzero(label_truth)
    rmse = sqrt(mean_squared_error(pred2[one_label_index], ground_truth[one_label_index]))
    mae = mean_absolute_error(pred2[one_label_index], ground_truth[one_label_index])
    i_auc, iPR_auc, acc, mcc = classification_metrics(label_truth, pred1)

    return i_auc, iPR_auc, rmse, mae, acc, mcc, ground_truth, pred1, raw_pred1, pred2


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
    parser.add_argument('--feature_mode', choices=['similarity', 'tfidf_svd'],
                        default='tfidf_svd', help='特征模式：原基座 similarity 或 TF-IDF-SVD')
    parser.add_argument('--use_rpu', action=argparse.BooleanOptionalAction, default=True,
                        help='统一控制 RPU 加权负采样与 logit adjustment')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        metavar='FLOAT', help='基础模型早停验证集比例')
    parser.add_argument('--use_topology_fusion', action=argparse.BooleanOptionalAction, default=True,
                        help='启用无泄露的线性拓扑校准')
    parser.add_argument('--topology_val_ratio', type=float, default=0.1,
                        metavar='FLOAT', help='拓扑校准模式下早停验证集占外层训练折的比例')
    parser.add_argument('--topology_calibration_ratio', type=float, default=0.1,
                        metavar='FLOAT', help='拓扑校准集占外层训练折的比例')
    parser.add_argument('--lambda_cls', type=float, default=0.7,
                        metavar='FLOAT', help='分类损失权重')
    parser.add_argument('--lambda_reg', type=float, default=0.2,
                        metavar='FLOAT', help='回归损失权重')
    args = parser.parse_args()
    set_random_seed(SEED)
    print(f"[FeatureMode] {args.feature_mode}")
    print(f"[TrainingStrategy] rpu_logit_adjustment={args.use_rpu}")
    print(f"[TopologyFusion] enabled={args.use_topology_fusion}")
    print(f"[Seed] global_rpu_split={SEED}, outer_fold={OUTER_FOLD_SEED}")
    configure_cpu_threads(
        args.torch_threads,
        args.torch_interop_threads,
        prefer_cuda=torch.cuda.is_available()
    )
    print(f"[LogitAdjustment] bias={compute_logit_adjustment_bias(args.use_rpu):.4f}")
    args.rawpath, args.similarity_path = ensure_training_feature_cache(#生成相似度特征
        args.rawpath,
        args.similarity_path
    )
    
    drug_ids, adr_ids, drug_side = load_label(args)
    print(f"[Dataset] drug_side shape={drug_side.shape}")

    (
        similarity_drug_features,
        similarity_side_features,
        similarity_drug_feature_names,
        similarity_side_feature_names,
    ) = load_similarity_features(drug_ids, adr_ids, args)

    tfidf_drug_dense_features = {}
    tfidf_side_dense_features = {}
    mesh_token_feature = None
    if args.feature_mode == "tfidf_svd":
        gene_svd = ensure_gene_tfidf_svd(
            args.rawpath, args.similarity_path, drug_ids, adr_ids
        )
        (
            tfidf_drug_dense_features,
            tfidf_side_dense_features,
            mesh_token_feature,
        ) = load_tfidf_svd_features(drug_ids, adr_ids, args, gene_svd)
    
    # 固定随机状态即可复现外层负样本与五折索引，不额外生成协议文件。
    final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values)
    data = np.vstack((final_positive_sample, final_negative_sample)).astype(np.float32)
    data_x = data[:, :2]
    # 五折分类分层只区分正负样本，原始 rating 仍完整保留在 data 中。
    data_y = (data[:, 2] > 0).astype(np.int32)
    
    # 正常五折训练
    fold = 1
    total_auc, total_pr_auc, total_rmse, total_mae = [], [], [], []
    total_acc, total_mcc = [], []
    #建立输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    normalized_rawpath = os.path.normpath(args.rawpath)
    output_dir = os.path.join(normalized_rawpath, f'output_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    #开始五折交叉验证
    kfold = StratifiedKFold(OUTER_FOLD_COUNT, random_state=OUTER_FOLD_SEED, shuffle=True)
    for train_split, test_split in kfold.split(data_x, data_y):
        # 每折从同一随机初始状态开始，RPU、DataLoader 与 CNN 初始化均可复现。
        set_random_seed(SEED)
        print("==================================fold {} start".format(fold))
        fold_train_data = data[train_split].tolist()
        test_data = data[test_split].tolist()
        calibration_data = []
        if args.use_topology_fusion:
            # 外层 train 折再拆成三个互斥部分：基础训练用于拟合 CNN，验证用于早停，
            # 校准用于学习最终线性排序器。外层 test 折直到最后评估前都完全不可见。
            fold_train_data, val_data, calibration_data = split_train_val_calibration(
                fold_train_data,
                val_ratio=args.topology_val_ratio,
                calibration_ratio=args.topology_calibration_ratio,
            )
            # 在构建任意特征前校验 pair 互斥；一旦重叠立即终止，避免静默泄露。
            assert_disjoint_pair_splits(
                fold_train_data,
                val_data,
                calibration_data,
                test_data,
            )
        else:
            fold_train_data, val_data = split_train_val(
                fold_train_data,
                val_ratio=args.val_ratio,
            )
        print(
            f"[Fold {fold}] train={len(fold_train_data)}, val={len(val_data)}, "
            f"calibration={len(calibration_data)}, test={len(test_data)}",
            flush=True
        )
        # 所有非基础训练 pair 都进入隐藏集合，后续 DSA、DSARaw、RPU 和拓扑传播统一屏蔽。
        # 这不仅屏蔽正边，也屏蔽验证/校准/测试中的负 pair，保证各阶段候选空间互不干扰。
        hidden_data = val_data + calibration_data + test_data
        topology_scorer = None
        if args.use_topology_fusion:
            # 拓扑 A 只接收 fold_train_data 的正边。药物 40 邻居、ADR 80 邻居分别固定，
            # 因为两侧实体数量和相似度稠密度不同，不能用同一个 top-k 强行约束。
            topology_scorer = build_topology_scorer(
                similarity_drug_features,
                similarity_side_features,
                fold_train_data,
                drug_side.shape,
                drug_topk=40,
                adr_topk=80,
            )
            assert_hidden_pairs_are_masked(topology_scorer.visible_matrix, hidden_data)
        if args.use_rpu or args.feature_mode == "similarity":
            (
                rpu_fold_drug_features,
                rpu_fold_side_features,
                rpu_fold_drug_feature_names,
                rpu_fold_side_feature_names,
            ) = build_fold_similarity_features(
                similarity_drug_features,
                similarity_side_features,
                similarity_drug_feature_names,
                similarity_side_feature_names,
                drug_side,
                hidden_data,
            )

        if args.use_rpu:
            # RPU 只从当前基础训练折可用的未观察 pair 中固定采样负样本。
            # 返回五列训练格式：实体索引、软标签、真实 rating 与负样本风险权重。
            fold_train_data = build_rpu_train_samples(
                fold_train_data,
                hidden_data,
                drug_side.values,
                rpu_fold_drug_features,
                rpu_fold_side_features,
                seed=SEED,
            )
        else:
            fold_train_data = to_weighted_train_samples(fold_train_data)

        if args.feature_mode == "similarity":
            prediction_drug_features = rpu_fold_drug_features
            prediction_side_features = rpu_fold_side_features
            prediction_drug_feature_names = rpu_fold_drug_feature_names
            prediction_side_feature_names = rpu_fold_side_feature_names
            prediction_mesh_token_feature = None
        else:
            (
                # TF-IDF/SVD 的实体属性特征与按 hidden_data 清零后的 DSARaw 一起使用。
                prediction_drug_features,
                prediction_side_features,
                prediction_drug_feature_names,
                prediction_side_feature_names,
            ) = build_tfidf_svd_prediction_features(
                tfidf_drug_dense_features,
                tfidf_side_dense_features,
                drug_side,
                hidden_data,
            )
            prediction_mesh_token_feature = mesh_token_feature

        print(
            f"[Prediction] Fold {fold} drug={prediction_drug_feature_names}, "
            f"adr={prediction_side_feature_names}"
        )
        auc, PR_auc, rmse, mae, acc, mcc = train_test(
            prediction_drug_features,
            prediction_side_features,
            fold_train_data,
            val_data,
            test_data,
            fold,
            args,
            output_dir,
            mesh_token_feature=prediction_mesh_token_feature,
            calibration_data=calibration_data,
            topology_scorer=topology_scorer,
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

    # 将最终五折均值写入结果文件，使文件内容与终端最后一组 Total_* 保持一致。
    summary = {
        "AUC": np.mean(total_auc),
        "AUPR": np.mean(total_pr_auc),
        "RMSE": np.mean(total_rmse),
        "MAE": np.mean(total_mae),
        "ACC": np.mean(total_acc),
        "MCC": np.mean(total_mcc),
    }
    with open(os.path.join(output_dir, 'results.txt'), 'a+') as f:
        f.write("\n===== Summary =====\n")
        for name, value in summary.items():
            f.write(f"{name}: {value:.10f}\n")
