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
from utils.data_utils import data_feature, jaccard_similarity, Convert_triplelist2matrix
# ChemProp 依赖已移除

# 设置随机种子确保可复现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def configure_cpu_threads(torch_threads, torch_interop_threads):
    """限制 PyTorch 的 CPU 线程池，避免训练时少数核心长时间满载。"""
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


def get_d4_contrastive_weight(epoch, args):
    """D4对比损失权重热身，降低训练早期假阴性梯度冲击。"""
    if not args.use_d4_contrastive_warmup:
        return args.contrastive_weight
    warmup_epochs = max(1, int(args.d4_warmup_epochs))
    return args.contrastive_weight * min(1.0, epoch / warmup_epochs)


def build_d4_drug_similarity(drug_features):
    """融合药物多源相似性矩阵，作为负样本假阴性风险的结构依据。"""
    normalized_sims = []
    for sim in drug_features:
        sim = np.nan_to_num(np.asarray(sim, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sim_min, sim_max = float(sim.min()), float(sim.max())
        if sim_min < 0.0 or sim_max > 1.0:
            sim = (sim - sim_min) / (sim_max - sim_min + 1e-12)
        normalized_sims.append(np.clip(sim, 0.0, 1.0))
    return np.mean(normalized_sims, axis=0)


def compute_d4_negative_risks(negative_samples, DAL, drug_sim):
    """计算每个未报告负样本靠近同ADR阳性药物的程度，值越大越像假阴性。"""
    negative_samples = np.asarray(negative_samples)
    side_ids = negative_samples[:, 1].astype(int)
    drug_ids = negative_samples[:, 0].astype(int)
    risks = np.zeros(len(negative_samples), dtype=np.float32)

    for side_idx in np.unique(side_ids):
        positive_drugs = np.flatnonzero(DAL[:, side_idx] > 0)
        if len(positive_drugs) == 0:
            continue
        sample_idx = np.flatnonzero(side_ids == side_idx)
        risks[sample_idx] = drug_sim[drug_ids[sample_idx]][:, positive_drugs].max(axis=1)

    return np.clip(np.nan_to_num(risks, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def d4_similarity_aware_negative_resampling(addition_negative_sample, final_positive_sample,
                                            final_negative_sample, DAL, drug_features, args):
    """复用剩余负样本池，对高假阴性风险负样本降权后重新抽取1:1负样本。"""
    if not args.use_d4_similarity_negative_weighting:
        return final_negative_sample

    candidate_negative = np.vstack((final_negative_sample, addition_negative_sample))
    drug_sim = build_d4_drug_similarity(drug_features)
    risks = compute_d4_negative_risks(candidate_negative, DAL, drug_sim)
    cutoff = np.percentile(risks, args.d4_negative_risk_percentile)

    # 高于分位阈值的负样本更可能是假阴性，降权但不直接删除，避免过度筛数据。
    weights = np.where(risks > cutoff, 1.0 - risks, 1.0)
    weights = np.clip(weights, args.d4_negative_min_weight, 1.0)
    probs = weights / weights.sum()

    sample_size = len(final_positive_sample)
    sampled_idx = np.random.choice(len(candidate_negative), size=sample_size, replace=False, p=probs)
    sampled_negative = candidate_negative[sampled_idx]
    sampled_risks = risks[sampled_idx]

    args.d4_negative_candidate_count = int(len(candidate_negative))
    args.d4_negative_sampled_count = int(len(sampled_negative))
    args.d4_negative_risk_cutoff = float(cutoff)
    args.d4_negative_risk_mean_before = float(risks.mean())
    args.d4_negative_risk_mean_after = float(sampled_risks.mean())

    print("[D4] similarity-aware negative weighting enabled")
    print(f"[D4] negative candidates: {len(candidate_negative)}, sampled negatives: {len(sampled_negative)}")
    print(f"[D4] risk percentile cutoff: {cutoff:.4f}, min_weight: {args.d4_negative_min_weight:.4f}")
    print(f"[D4] risk mean before/after: {risks.mean():.4f} / {sampled_risks.mean():.4f}")
    return sampled_negative


# 设置系统路径
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path + "/..")

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_label(screen_drug_list, use_DGen, use_AGen, args):
    """Load and construct drug-side effect label matrix.
    
    Args:
        screen_drug_list: List of drug IDs to screen
        use_DGen: Whether to filter by drug-gene interactions
        use_AGen: Whether to filter by gene-disease associations
        args: Command line arguments
        
    Returns:
        drug_list: List of drug IDs
        adr_list: List of side effect IDs
        drug_side: Drug-side effect interaction matrix
    """
    drug_col, adr_col = "pert_id", "MESH_ID"
    
    # Check cache
    cache_filename = "drug_side.csv"
    cache_path = os.path.join(args.similarity_path, cache_filename)
    
    if os.path.exists(cache_path):
        print(f"[Cache] Loading from cache: {cache_path}")
        drug_side = pd.read_csv(cache_path, header=0, index_col=0)
        return list(drug_side.index), list(drug_side.columns), drug_side

    # Load label data
    pd_label = pd.read_csv(args.rawpath + "sider_pert_mesh_list.csv", header=0, delimiter='\t')
    drug_side = data_feature(pd_label, screen_list=screen_drug_list, 
                             screen_col=drug_col, del_screen_col=False)
    
    # Filter by drug-gene interactions
    if use_DGen:
        pd_DGen = pd.read_csv(args.rawpath + "ctd_chem_pert_gene_ixns_list.csv", 
                              header=0, delimiter='\t')
        drug_list_DGen = sorted(np.unique(pd_DGen[drug_col]).tolist())
        drug_side = pd.concat([drug_side[drug_side[drug_col] == id] 
                               for id in drug_list_DGen], ignore_index=True)
    
    # Filter by gene-disease associations
    if use_AGen:
        pd_AGen = pd.read_csv(args.rawpath + "ctd_gene_adr_asso_list_4386.csv", 
                              header=0, delimiter='\t')
        adr_list_Gen = sorted(np.unique(pd_AGen[adr_col]).tolist())
        drug_side = pd.concat([drug_side[drug_side[adr_col] == id] 
                               for id in adr_list_Gen], ignore_index=True)

    # Convert to matrix format
    drug_side = Convert_triplelist2matrix(drug_side, ["pert_id", "MESH_ID", "label"], 
                                           fillna_val=0)
    
    # Save to cache
    try:
        os.makedirs(args.rawpath, exist_ok=True)
        drug_side.to_csv(cache_path, header=True, index=True)
        print(f"[Cache] Saved label matrix to: {cache_path}")
    except Exception as e:
        print(f"[Cache] Save failed: {e}")
    
    return list(drug_side.index), list(drug_side.columns), drug_side 

def load_drug_feature(screen_drug_list, args):
    """加载药物相似性特征矩阵。
    
    特征包括:
    - 药物-基因相互作用 (DGen)
    - 基因表达 (GE) 
    - 化学结构 (CS)
    
    Args:
        screen_drug_list: 药物ID列表
        args: 命令行参数
        
    Returns:
        药物相似性矩阵列表
    """
    print(f"\n{'='*60}")
    print("加载药物特征")
    print(f"{'='*60}")
    
    # 加载预先计算的相似性矩阵
    drug_cs = np.array(pd.read_csv(os.path.join(args.similarity_path, "drug_rdkit.csv"), header=0, index_col=0))
    drug_DGen = np.array(pd.read_csv(os.path.join(args.similarity_path, "drug_DGen_sim.csv"), header=0, index_col=0))
    drug_ge_sim = np.array(pd.read_csv(os.path.join(args.similarity_path, "drug_ge_sim.csv"), header=0, index_col=0))
    
    # 聚合特征
    drug_features = [drug_DGen, drug_ge_sim, drug_cs]
    
    print(f"\n药物特征已加载:")
    print(f"  - DGen:   {drug_DGen.shape}")
    print(f"  - GE:     {drug_ge_sim.shape}")
    print(f"  - CS:     {drug_cs.shape}")
    print(f"{'='*60}\n")
    
    return drug_features

def load_adr_feature(screen_adr_list, args):
    """Load side effect similarity features.
    
    Features include:
    - MESH ontology similarity
    - Gene-Disease associations (GDA)
    
    Args:
        screen_adr_list: List of side effect IDs
        args: Command line arguments
        
    Returns:
        List of side effect similarity matrices
    """
    print(f"\n{'='*60}")
    print("Loading Side Effect Features")
    print(f"{'='*60}")
    
    # Load precomputed similarity matrices
    side_mesh_sim = np.array(pd.read_csv(os.path.join(args.similarity_path, "side_mesh_sim.csv"), 
                                          header=0, index_col=0))
    adr_GDisease_sim = np.array(pd.read_csv(os.path.join(args.similarity_path, "adr_GDisease_sim.csv"), 
                                             header=0, index_col=0))
    
    # Aggregate features
    side_features = [side_mesh_sim, adr_GDisease_sim]
    
    print(f"\nSide effect features loaded:")
    print(f"  - MESH: {side_mesh_sim.shape}")
    print(f"  - GDA:  {adr_GDisease_sim.shape}")
    print(f"{'='*60}\n")
    
    return side_features    

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
    interaction_target = np.zeros((n_samples, 3), dtype=int)#[num_drug*num_adr, 3]
    
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




# ============================================================================
# Training and Evaluation Functions
# ============================================================================

def train_test(drug_feature, side_feature, data_train, data_test, fold, args, remain_drug_list, adr_list, output_dir):
    """一折的训练和评估函数。
    
    Args:
        drug_feature: 药物相似性矩阵列表
        side_feature: 副作用相似性矩阵列表
        data_train: 训练样本
        data_test: 测试样本
        fold: 当前折数
        args: 命令行参数
        remain_drug_list: 药物ID列表
        
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
    data_test = np.array(data_test)
    
    train_indices = (
        data_train[:, 0].astype(int),  # drug_indices
        data_train[:, 1].astype(int),  # side_indices
        data_train[:, 2]               # labels
    )
    
    test_indices = (
        data_test[:, 0].astype(int),
        data_test[:, 1].astype(int),
        data_test[:, 2]
    )
    
    '''构建训练集和测试集'''
    trainset = torch.utils.data.TensorDataset(
        torch.LongTensor(train_indices[0]),  # drug_indices
        torch.LongTensor(train_indices[1]),  # side_indices
        torch.FloatTensor(train_indices[2])  # labels
    )
    testset = torch.utils.data.TensorDataset(
        torch.LongTensor(test_indices[0]),
        torch.LongTensor(test_indices[1]),
        torch.FloatTensor(test_indices[2])
    )
    
    _test = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True,
                                        num_workers=0, pin_memory=True)

    _train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, pin_memory=True)
    
    '''配置cuda加速'''
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # ChemProp 编码器已移除
    
    '''构建模型'''
    n_drug_chunks = len(drug_feature)
    n_side_chunks = len(side_feature)
    d4_method = args.contrastive_loss_type
    if args.use_d4_contrastive_warmup:
        d4_method += "+warmup"
    if args.use_d4_contrastive_scale_norm:
        d4_method += "+scale_norm"
    if args.use_d4_similarity_negative_weighting:
        d4_method += "+similarity_negative_weighting"
    print(f"D4 contrastive method: {d4_method}")
    model = DGAPred(
        drugs_dim=drug_feature[0].shape[0]*n_drug_chunks,
        sides_dim=side_feature[0].shape[0]*n_side_chunks,
        embed_dim=args.embed_dim,
        batchsize=args.batch_size,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        n_drug_chunks=n_drug_chunks,
        n_side_chunks=n_side_chunks,
        use_feature_interaction=args.use_feature_interaction,
        use_contrastive_learning=args.use_contrastive_learning,
        contrastive_loss_type=args.contrastive_loss_type,
        d4_tau_plus=args.d4_tau_plus
    ).to(device)
    
    '''构建损失函数和优化器'''
    Regression_criterion = nn.MSELoss()
    Classification_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    '''构建学习率调度器'''
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    '''初始化训练指数变量'''
    AUC_mn = 0
    AUPR_mn = 0

    rms_mn = 100000
    mae_mn = 100000
    endure_count = 0
    best_model_state = None  # Save best model state

    start = time.time()
    train_epoches = []
    test_epoches = []
    
    '''训练'''
    for epoch in range(1, args.epochs + 1):
        iter_loss_sum, step = train(model, _train_loader, optimizer, Classification_criterion, Regression_criterion, device, 
                                    global_drug_features_tensor, global_side_features_tensor, epoch, args) # 一个iterater
        train_epoch = iter_loss_sum/step
        train_epoches.append(train_epoch)
        
        # 清理CUDA缓存，防止显存碎片累积
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        t_i_auc, t_iPR_auc, t_rmse, t_mae, t_acc, t_mcc, t_ground_i, t_ground_u, t_ground_truth, t_pred1, t_pred2, test_iter_loss, test_step = test(model,
                                                                                                           _test,
                                                                                                           device,
                                                                                                           global_drug_features_tensor,
                                                                                                           global_side_features_tensor,
                                                                                                           lossfunction1=Classification_criterion,
                                                                                                           lossfunction2=Regression_criterion,
                                                                                                           epoch=epoch)
                                                                                        
        test_epoch = test_iter_loss/test_step
        test_epoches.append(test_epoch)


        is_better = (t_i_auc > AUC_mn) or (t_iPR_auc > AUPR_mn)
        if is_better:
            AUC_mn = max(AUC_mn, t_i_auc)
            AUPR_mn = max(AUPR_mn, t_iPR_auc)
            rms_mn = min(rms_mn, t_rmse)
            mae_mn = min(mae_mn, t_mae)
            endure_count = 0
            # Save best model state
            best_model_state = deepcopy(model.state_dict())
        else:
            endure_count += 1

        if scheduler is not None:
            scheduler.step(t_i_auc)

        print("Epoch: %d <Test after train-epoch> RMSE: %.5f, MAE: %.5f, AUC: %.5f, AUPR: %.5f, ACC: %.5f, MCC: %.5f " % (
        epoch, t_rmse, t_mae, t_i_auc, t_iPR_auc, t_acc, t_mcc))
        start = time.time()

        if endure_count >15 :
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
        epoch=epoch
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
            f.write(f"D4 contrastive method: {d4_method}\n")
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
          global_drug_features, global_side_features, epoch, args=None):
    """训练函数 - 带进度条和实时指标"""
    model.train()
    
    avg_loss = 0.0
    losses = []  # 记录每个batch的loss

    # 创建进度条
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc="Training")
    for step, (drug_idx, side_idx, ratings) in pbar:
        
        # 构建二分类标签
        labels = (ratings > 0).float()
        
        optimizer.zero_grad()
        
        # 前向传播
        model_output = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            epoch=epoch  # ARConv 需要 epoch 参数来自适应调整卷积核
        )
        #已写完
        # Handle contrastive learning output
        if len(model_output) == 3:
            logits, reconstruction, contrastive_loss = model_output
        else:
            logits, reconstruction = model_output
            contrastive_loss = None
        
        one_label_index = np.nonzero(labels.data.numpy())
        
        # 标签平滑
        if args.label_smooth > 0:
            eps = float(args.label_smooth)
            y_target = (1.0 - eps) * labels + 0.5 * eps
        else:
            y_target = labels
        
        # 计算损失
        loss1 = lossfunction1(logits, y_target.to(device))
        loss2 = lossfunction2(reconstruction[one_label_index], ratings[one_label_index].to(device))
        lambda_cls = 0.7  # 分类任务权重
        total_loss = lambda_cls * loss1 + (1 - lambda_cls) * loss2
        
        # D4：对比损失可选去偏、权重热身和尺度归一化，不改变评估逻辑。
        if contrastive_loss is not None:
            contrastive_weight = get_d4_contrastive_weight(epoch, args)
            if args.use_d4_contrastive_scale_norm:
                scale_ref = loss1.detach().clamp(min=1e-3)
                cl_scale = contrastive_loss.detach().clamp(min=1e-3)
                contrastive_loss = contrastive_loss * (scale_ref / cl_scale)
            total_loss = total_loss + contrastive_weight * contrastive_loss
        
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

    return avg_loss, step

def test(model, test_loader, device, global_drug_features, global_side_features, lossfunction1, lossfunction2, epoch=0):
    """测试函数 - 带进度条和实时指标"""
    model.eval()
    
    pred1 = []
    pred2 = []
    ground_truth = []
    label_truth = []
    ground_u = []
    ground_i = []
    test_avg_loss = 0.0
    
    # 创建进度条，使用no_grad避免构建计算图
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    with torch.no_grad():
      for step, (drug_idx, side_idx, ratings) in pbar:
        # 构建二分类标签
        labels = (ratings > 0).float()
        
        ground_i.append(list(drug_idx.data.cpu().numpy()))
        ground_u.append(list(side_idx.data.cpu().numpy()))
        
        # 前向传播score_one:classfication score_two:regression
        scores_one, scores_two = model(
            drug_indices=drug_idx,
            side_indices=side_idx,
            device=device,
            global_drug_features=global_drug_features,
            global_side_features=global_side_features,
            epoch=epoch  # ARConv 需要 epoch 参数
        )
        one_label_index = np.nonzero(labels.data.numpy())
        
        # 计算损失
        loss1 = lossfunction1(scores_one, labels.to(device))#BCEWithLogitsLoss内部会做sigmoid
        loss2 = lossfunction2(scores_two[one_label_index], ratings[one_label_index].to(device))#在正样本上计算MSELoss
        lambda_cls = 0.7
        test_loss = lambda_cls * loss1 + (1 - lambda_cls) * loss2
        test_avg_loss += test_loss.detach().item()
        
        # 收集预测结果
        prob_one = torch.sigmoid(scores_one)
        pred1.append(list(prob_one.data.cpu().numpy()))
        pred2.append(list(scores_two.data.cpu().numpy()))
        ground_truth.append(list(ratings.data.cpu().numpy()))
        label_truth.append(list(labels.data.cpu().numpy()))
        
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

    return i_auc, iPR_auc, rmse, mae, acc, mcc, ground_i, ground_u, ground_truth, pred1, pred2, test_avg_loss, step


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description = 'Model')
    parser.add_argument('--epochs', type = int, default = 120,
                        metavar = 'N', help = 'number of epochs to train')
    parser.add_argument('--lr', type = float, default = 1e-3,
                        metavar = 'FLOAT', help = 'learning rate')
    parser.add_argument('--embed_dim', type = int, default = 128,
                        metavar = 'N', help = 'embedding dimension')
    parser.add_argument('--weight_decay', type = float, default = 1e-5,
                        metavar = 'FLOAT', help = 'weight decay')
    parser.add_argument('--batch_size', type = int, default = 128,
                        metavar = 'N', help = 'input batch size for training')
    parser.add_argument('--test_batch_size', type = int, default =128,
                        metavar = 'N', help = 'input batch size for testing')
    parser.add_argument('--torch_threads', type=int, default=0,
                        metavar='N', help='PyTorch CPU算子线程数，0表示使用环境默认值')
    parser.add_argument('--torch_interop_threads', type=int, default=0,
                        metavar='N', help='PyTorch CPU算子调度线程数，0表示使用环境默认值')
    parser.add_argument('--rawpath', type=str, default='pythonPredict/DGAPred(Compare)/2drug-2side/DGAPred/data/',
                        metavar='STRING', help='rawpath')

    parser.add_argument('--similarity_path', type=str, default='pythonPredict/',
                        metavar='STRING', help='similarity matrices path')
    # 训练稳健性与正则化
    parser.add_argument('--dropout1', type=float, default=0.4,metavar='FLOAT', help='主特征编码阶段的dropout')
    parser.add_argument('--dropout2', type=float, default=0.2,metavar='FLOAT', help='Final prediction dropout rate')
    parser.add_argument('--label_smooth', type=float, default=0.05,metavar='FLOAT', help='二分类标签平滑系数(0~0.2)，仅训练使用')
    parser.add_argument('--grad_clip', type=float, default=0.5,metavar='FLOAT', help='梯度裁剪阈值，<=0 关闭')
    parser.add_argument('--use_scheduler', action='store_true', help='启用基于验证AUC的ReduceLROnPlateau学习率调度',default=True)
    # FIA-DTA 2025: 特征交互注意力机制，增强药物与靶标的多模态特征交互
    # CCL-ASPS 2024: 协同对比学习框架，提升模型表征能力
    parser.add_argument('--use_feature_interaction', action='store_true', help='启用特征交互注意力 (FIA-DTA 2025)',default=True)
    parser.add_argument('--use_contrastive_learning', action='store_true', help='启用协同对比学习 (CCL-ASPS 2024)',default=True)
    parser.add_argument('--contrastive_weight', type=float, default=0.20, metavar='FLOAT', help='对比学习损失权重')
    parser.add_argument('--contrastive_loss_type', type=str, default='standard',
                        choices=['standard', 'debiased'], help='D4对比损失类型')
    parser.add_argument('--d4_tau_plus', type=float, default=0.10,
                        metavar='FLOAT', help='D4去偏InfoNCE假阴性比例先验')
    parser.add_argument('--use_d4_contrastive_warmup', action='store_true',
                        help='启用D4对比损失权重热身')
    parser.add_argument('--d4_warmup_epochs', type=int, default=15,
                        metavar='N', help='D4对比损失权重热身轮数')
    parser.add_argument('--use_d4_contrastive_scale_norm', action='store_true',
                        help='启用D4对比损失尺度归一化')
    parser.add_argument('--use_d4_similarity_negative_weighting', action='store_true',
                        help='启用D4相似性风险负样本降权重采样')
    parser.add_argument('--d4_negative_risk_percentile', type=float, default=90.0,
                        metavar='FLOAT', help='D4高假阴性风险负样本分位阈值')
    parser.add_argument('--d4_negative_min_weight', type=float, default=0.05,
                        metavar='FLOAT', help='D4高风险负样本最低保留权重')

    args = parser.parse_args()
    configure_cpu_threads(args.torch_threads, args.torch_interop_threads)
    druglist = pd.read_csv(args.rawpath+"lincs_druglist_ge_go_521.csv")
    
    remain_drug_list,adr_list, drug_side = load_label(druglist["pert_id"],True,True, args)#drug_sided的行索引remain_drug_list,列索引adr_list
    print("drug_side shape:",pd.DataFrame(drug_side).shape)

    # 加载药物和不良反应特征；D4负样本风险评分需要先拿到药物相似性矩阵。
    drug_feature = load_drug_feature(remain_drug_list, args)
    side_feature = load_adr_feature(adr_list, args)
    
    #不参与训练的负样本，len(final_positive_sample)=len(final_negative_sample)
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(drug_side.values, addition_negative_number='all')#分离正负样本并均衡正负样本
    final_negative_sample = d4_similarity_aware_negative_resampling(
        addition_negative_sample,
        final_positive_sample,
        final_negative_sample,
        drug_side.values,
        drug_feature,
        args
    )
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
        auc, PR_auc, rmse, mae, acc, mcc = train_test(drug_feature,side_feature,data[train_split].tolist(), data[test_split].tolist(),fold,args,remain_drug_list,adr_list,output_dir)
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
