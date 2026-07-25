"""无泄露的线性多尺度拓扑排序校准。"""

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def classification_metrics(labels, probabilities):
    """计算分类任务使用的四项指标。"""
    precision, recall, _ = metrics.precision_recall_curve(labels, probabilities, pos_label=1)
    predictions = (probabilities >= 0.5).astype(np.int32)
    return (
        metrics.roc_auc_score(labels, probabilities),
        metrics.auc(recall, precision),
        metrics.accuracy_score(labels, predictions),
        metrics.matthews_corrcoef(labels, predictions),
    )


def _rank_pairs(labels):
    """为校准集内每个正样本固定生成八个负样本配对用于排序损失"""
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels <= 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("排序校准需要同时包含正样本和负样本。")
    # 局部随机生成器固定为 42，不受 CNN 训练或 RPU 采样消耗随机状态的影响。
    # 若校准集中有 P 个正样本，返回的两个索引数组长度均为 8P；每个位置是一对正负样本。
    # 这些索引只指向当前校准集，外层测试集从未参与配对或权重学习。
    choices = np.random.default_rng(42).choice(negative, size=(len(positive), 8), replace=True)#抽样得到(len(positive), 8)size
    return np.repeat(positive, 8), choices.reshape(-1)


def fit_topology_calibration(features, labels):
    """在独立校准集拟合标准化线性 BCE 加 RankNet 校准器。"""
    # features 形状为 (N, 17)：第一列是 CNN 原始 logit，其余 16 列是折内拓扑特征。
    # 不引入隐藏层，待学习参数只有 weight(17,) 与标量 bias，因此每个权重可直接解释。
    # 标准化参数只在独立校准集拟合，并与线性权重一起保存到检查点。
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features).astype(np.float32)
    positive, negative = _rank_pairs(labels)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels.astype(np.float32))
    positive = torch.from_numpy(positive)
    negative = torch.from_numpy(negative)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    # 参数仅有一组线性权重和偏置。校准集较小且使用全量目标，L-BFGS 比逐批随机优化更稳定。
    optimizer = torch.optim.LBFGS(
        [weight, bias], lr=1.0, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        score = x @ weight + bias
        # BCE 保留概率拟合；RankNet 直接拉开正样本与其配对负样本的分数差。
        # softplus(-(s_pos-s_neg)) 在正样本分数高于负样本时趋近于零，故直接优化排序方向。
        bce = functional.binary_cross_entropy_with_logits(score, y)
        rank = functional.softplus(-(score[positive] - score[negative])).mean()#ln(1+e^x)排序损失 正样本分>负样本分
        # 轻量 L2 仅约束权重，避免小校准集产生过大的单特征系数。
        loss = bce + rank + 5e-5 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "mean": torch.as_tensor(scaler.mean_, dtype=torch.float32),
        "scale": torch.as_tensor(scaler.scale_, dtype=torch.float32),
        "weight": weight.detach(),
        "bias": bias.detach(),
    }


def predict_topology_calibration(parameters, features):
    """用保存的标准化参数和线性权重计算正类概率。"""
    # 推理阶段不能重新在测试集拟合 scaler；必须复用校准集保存的 mean 和 scale，
    # 否则测试集分布会反向影响预处理过程，且模型检查点无法独立复现。
    mean = parameters["mean"].cpu().numpy()
    scale = parameters["scale"].cpu().numpy()
    weight = parameters["weight"].cpu().numpy()
    bias = parameters["bias"].item()
    score = ((features - mean) / scale) @ weight + bias
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32)
