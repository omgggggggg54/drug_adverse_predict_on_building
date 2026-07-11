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
from utils.embedding_feature_generation import (
    ensure_adr_glove_similarity,
    ensure_drug_embedding_similarity,
    ensure_drug_unimol_similarity,
)
from utils.feature_generation import ensure_training_feature_cache, read_ordered_square_feature
from utils.rpu_utils import build_rpu_train_samples, to_weighted_train_samples
# ChemProp 依赖已移除

# 本地模型默认放在项目根目录，使用绝对路径避免从其他目录启动脚本时找不到模型。
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CHEMBERTA_MODEL_PATH = os.path.join(PROJECT_ROOT, "ChemBERTa")
DEFAULT_UNIMOL_MODEL_PATH = os.path.join(PROJECT_ROOT, "Uni-Mol")
DEFAULT_GLOVE_PATH = r"D:\learning\buliangfanying\数据集\glove.6B.300d.txt"

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
        ("GE", "drug_ge_sim.csv"),
        ("CS", "drug_rdkit.csv"),
        ("Morgan", "drug_morgan.csv"),
        ("MACCS", "drug_maccs.csv"),
        ("ChemBERTa", "drug_chemberta_sim.csv"),
        ("Uni-Mol", "drug_unimol_sim.csv"),
    ]

    drug_features = []
    drug_feature_names = []
    print(f"\n药物特征已加载:")
    for name, filename in feature_files:
        if name.upper() not in selected:
            print(f"  - {name}: skipped")
            continue
        if name == "ChemBERTa":
            ensure_drug_embedding_similarity(
                args.similarity_path,
                drug_ids,
                args.chemberta_model_path,
                filename,
                name,
            )
        elif name == "Uni-Mol":
            ensure_drug_unimol_similarity(
                args.similarity_path,
                drug_ids,
                args.unimol_model_path,
                filename,
                name,
            )
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
        ("GloVe", "adr_glove_sim.csv"),
    ]

    side_features = []
    side_feature_names = []
    print(f"\nADR特征已加载:")
    for name, filename in feature_files:
        if name.upper() not in selected:
            print(f"  - {name}: skipped")
            continue
        if name == "GloVe":
            ensure_adr_glove_similarity(
                args.similarity_path,
                adr_ids,
                args.glove_path,
                filename,
                name,
            )
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
    return {item.strip().upper() for item in str(text).split(",") if item.strip()}


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

def Extract_positive_negative_samples(DAL, addition_negative_number='all'):
    """Extract and balance positive and negative samples.
    
    Args:
        DAL: Drug-ADR label matrix
        addition_negative_number: Number of additional negative samples ('all' or int)
        
    Returns:
        addition_negative_sample: Extra negative samples
        final_positive_sample: Positive samples
        final_negative_sample: Balanced negative samples
    """
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
    
    # Sample balanced negatives
    neg_indices = list(range(number_negative))
    if addition_negative_number == 'all':
        sampled_indices = random.sample(neg_indices, number_negative)
    else:
        sampled_indices = random.sample(neg_indices, 
                                        (1 + addition_negative_number) * number_positive)
    
    final_negative_sample = negative_sample[sampled_indices[:number_positive]]
    addition_negative_sample = negative_sample[sampled_indices[number_positive:]]
    
    return addition_negative_sample, final_positive_sample, final_negative_sample


def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    """Sparse multi-label categorical cross-entropy loss (PyTorch implementation).
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted logits
        mask_zero: Whether to mask zero labels
        
    Returns:
        Combined positive and negative loss
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    return neg_loss + pos_loss


def compute_logit_adjustment_bias(args):
    """按RPU负采样比例做类别先验校正，让0.5阈值重新对应评估集的近似1:1先验。"""
    if args is None or not getattr(args, "use_logit_adjustment", False):
        return 0.0
    if getattr(args, "logit_adjustment_bias", None) is not None:
        return float(args.logit_adjustment_bias)
    if not getattr(args, "use_rpu", False):
        return 0.0
    ratio = max(float(getattr(args, "rpu_negative_ratio", 1.0)), 1e-6)
    return float(args.logit_adjustment_strength) * float(np.log(ratio))


# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_test(drug_feature, side_feature, data_train, data_val, data_test, fold, args, output_dir):
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
    drug_features_matrix_global = drug_feature[0]
    for i in range(1, len(drug_feature)):
        drug_features_matrix_global = np.hstack((drug_features_matrix_global, drug_feature[i]))
    
    side_features_matrix_global = side_feature[0]
    for i in range(1, len(side_feature)):
        side_features_matrix_global = np.hstack((side_features_matrix_global, side_feature[i]))
    
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
    trainset = torch.utils.data.TensorDataset(
        torch.LongTensor(train_indices[0]),  # drug_indices
        torch.LongTensor(train_indices[1]),  # side_indices
        torch.FloatTensor(train_indices[2]),  # soft_label
        torch.FloatTensor(train_indices[3]),  # rating
        torch.FloatTensor(train_indices[4])   # sample_weight
    )
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
        side_feature_dims=side_feature_dims
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
            Classification_criterion,
            Regression_criterion,
            device,
            global_drug_features_tensor,
            global_side_features_tensor,
            args=args
        )  # 一个iterater
        train_epoch = iter_loss_sum/step
        train_epoches.append(train_epoch)
        
        v_i_auc, v_iPR_auc, v_rmse, v_mae, v_acc, v_mcc, v_ground_i, v_ground_u, v_ground_truth, v_pred1, v_pred2, val_iter_loss, val_step = test(model,
                                                                                                           _val,
                                                                                                           device,
                                                                                                           global_drug_features_tensor,
                                                                                                           global_side_features_tensor,
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
    final_start = time.time()
    i_auc, iPR_auc, rmse, mae, acc, mcc, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step_ = test(
        model, _test, device, global_drug_features_tensor, global_side_features_tensor,
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

def train(model, train_loader, optimizer, lossfunction1, lossfunction2, device,
          global_drug_features, global_side_features, args=None):
    """训练函数 - 带进度条和实时指标"""
    model.train()
    
    avg_loss = 0.0
    losses = []  # 记录每个batch的loss

    # 创建进度条
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc="Training")
    for step, (drug_idx, side_idx, soft_labels, ratings, sample_weights) in pbar:
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
            global_side_features=global_side_features
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

        # 分类分支只保留基础 RPU 负样本加权。
        raw_bce = nn.functional.binary_cross_entropy_with_logits(logits, y_target, reduction='none')
        loss1 = (raw_bce * sample_weights).mean()

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

def test(model, test_loader, device, global_drug_features, global_side_features, lossfunction1, lossfunction2, args=None):
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
            global_side_features=global_side_features
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
    parser.add_argument('--chemberta_model_path', type=str, default=DEFAULT_CHEMBERTA_MODEL_PATH,
                        metavar='STRING', help='本地 ChemBERTa HuggingFace 模型目录')
    parser.add_argument('--unimol_model_path', type=str, default=DEFAULT_UNIMOL_MODEL_PATH,
                        metavar='STRING', help='本地 Uni-Mol 权重目录，需包含 mol_pre_no_h_220816.pt 和 mol.dict.txt')
    parser.add_argument('--glove_path', type=str, default=DEFAULT_GLOVE_PATH,
                        metavar='STRING', help='本地 GloVe 词向量文件路径')
    parser.add_argument('--drug_features', type=str, default='DGen,CS,DSA,ChemBERTa,Morgan',
                        help='药物特征列表，逗号分隔，例如 DGen,GE,CS,Morgan,MACCS,ChemBERTa,Uni-Mol,DSA')
    parser.add_argument('--adr_features', type=str, default='MESH,GDA,DSA',
                        help='ADR特征列表，逗号分隔，例如 MESH,GDA,DSA,GloVe')
    parser.add_argument('--val_source', choices=['train', 'test'], default='train',
                        help='验证集来源：train为严格口径，test仅用于复现旧口径')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        metavar='FLOAT', help='验证集划分比例')
    parser.add_argument('--seed', type=int, default=SEED,
                        metavar='N', help='RPU负采样随机种子')
    parser.add_argument('--lambda_cls', type=float, default=0.7,
                        metavar='FLOAT', help='分类损失权重')
    parser.add_argument('--lambda_reg', type=float, default=0.2,
                        metavar='FLOAT', help='回归损失权重')
    parser.add_argument('--use_logit_adjustment',default=True, action='store_true',
                        help='启用RPU类别先验logit校正，修正负采样导致的概率整体偏低')
    parser.add_argument('--logit_adjustment_strength', type=float, default=1.0,
                        metavar='FLOAT', help='log(rpu_negative_ratio)校正强度，1.0表示完整先验校正')
    parser.add_argument('--logit_adjustment_bias', type=float, default=None,
                        metavar='FLOAT', help='手动指定logit校正偏置；设置后覆盖自动计算')
    parser.add_argument('--use_rpu',default=True, action='store_true',
                        help='启用RPU-DGAPred E3训练样本构造')
    parser.add_argument('--rpu_negative_mode', choices=['all_weighted'], default='all_weighted',
                        help='RPU负样本模式，E3固定为all_weighted')
    parser.add_argument('--rpu_negative_ratio', type=int, default=10,
                        metavar='N', help='每个训练正样本对应的未观察负样本数量')
    parser.add_argument('--rpu_min_neg_weight', type=float, default=0.2,
                        metavar='FLOAT', help='高风险未观察负样本的最低分类权重')
    parser.add_argument('--rpu_drug_risk_weight', type=float, default=0.7,
                        metavar='FLOAT', help='drug-side风险在总风险中的权重')
    parser.add_argument('--rpu_use_consensus_pseudo',default=True, action='store_true',
                        help='启用多视图共识伪阳性：多个特征视图一致高分的未知pair会以弱标签加入训练')
    parser.add_argument('--rpu_consensus_pseudo_ratio', type=float, default=0.1,
                        metavar='FLOAT', help='伪阳性数量占当前fold真实正样本数的比例，count为0时生效')
    parser.add_argument('--rpu_consensus_pseudo_count', type=int, default=0,
                        metavar='N', help='每个fold固定加入的伪阳性数量；0表示按ratio自动计算')
    parser.add_argument('--rpu_consensus_threshold', type=float, default=0.85,
                        metavar='FLOAT', help='多视图平均分和单视图同意的高置信阈值')
    parser.add_argument('--rpu_consensus_min_view_score', type=float, default=0.65,
                        metavar='FLOAT', help='入选伪阳性时每个视图允许的最低分，防止单视图明显反对')
    parser.add_argument('--rpu_consensus_max_std', type=float, default=0.08,
                        metavar='FLOAT', help='多视图分数标准差上限，越小表示共识越严格')
    parser.add_argument('--rpu_consensus_min_agree_views', type=int, default=2,
                        metavar='N', help='至少多少个视图需要达到consensus_threshold')
    parser.add_argument('--rpu_min_pseudo_label', type=float, default=0.6,
                        metavar='FLOAT', help='伪阳性soft label下限')
    parser.add_argument('--rpu_max_pseudo_label', type=float, default=0.95,
                        metavar='FLOAT', help='伪阳性soft label上限')
    parser.add_argument('--rpu_min_pseudo_weight', type=float, default=0.3,
                        metavar='FLOAT', help='伪阳性分类损失权重下限')
    parser.add_argument('--rpu_max_pseudo_weight', type=float, default=0.8,
                        metavar='FLOAT', help='伪阳性分类损失权重上限')
    args = parser.parse_args()
    configure_cpu_threads(
        args.torch_threads,
        args.torch_interop_threads,
        prefer_cuda=torch.cuda.is_available()
    )
    if args.use_logit_adjustment:
        print(f"[LogitAdjustment] bias={compute_logit_adjustment_bias(args):.4f}, "
              f"strength={args.logit_adjustment_strength}")
    args.rawpath, args.similarity_path = ensure_training_feature_cache(
        args.rawpath,
        args.similarity_path
    )
    
    drug_ids, adr_ids, drug_side = load_label(args)
    print("drug_side shape:",pd.DataFrame(drug_side).shape)

    # 加载药物和不良反应特征；读取时再次校验顺序，防止特征矩阵与标签矩阵错位。
    drug_feature, drug_feature_names = load_drug_feature(drug_ids, args)
    side_feature, side_feature_names = load_adr_feature(adr_ids, args)
    
    # 外层交叉验证仍使用1:1正负样本，保证验证/测试口径和原DGAPred对照一致。
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values, addition_negative_number='all')
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
        fold_drug_feature, fold_side_feature, fold_drug_feature_names, fold_side_feature_names = add_dsa_features(
            drug_feature,
            side_feature,
            drug_feature_names,
            side_feature_names,
            drug_side,
            val_data + test_data,
            args
        )
        if args.use_rpu:
            fold_train_data = build_rpu_train_samples(
                fold_train_data,
                val_data + test_data,
                drug_side.values,
                fold_drug_feature,
                fold_side_feature,
                args,
                fold,
                fold_drug_feature_names,
                fold_side_feature_names,
            )
        else:
            fold_train_data = to_weighted_train_samples(fold_train_data)
        auc, PR_auc, rmse, mae, acc, mcc = train_test(
            fold_drug_feature,
            fold_side_feature,
            fold_train_data,
            val_data,
            test_data,
            fold,
            args,
            output_dir
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
