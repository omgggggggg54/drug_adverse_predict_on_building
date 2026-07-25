"""严格基于基础训练正边构造多尺度双拓扑特征。"""

from dataclasses import dataclass

import numpy as np


# 此顺序与 build_topology_scorer 中堆叠后的最后一维严格一致。
# main.py 会在最前面拼接 CNN 原始 logit，因此线性校准器的完整输入为：
# [cnn_logit, 以下 16 个特征]，总维度为 17。
MULTISCALE_FEATURE_NAMES = [
    # 0~1：基础训练正边在两个实体侧的度数。使用 log1p 压缩高度节点的数值范围。
    "drug_degree",
    "adr_degree",
    # 2~5：从药物或 ADR 相似图传播一次得到的关联强度。
    "DGen_1hop",
    "CS_1hop",
    "MESH_1hop",
    "GDA_1hop",
    # 6~9：在同一相似图中继续传播一次，刻画更远的同侧邻居。
    "DGen_2hop",
    "CS_2hop",
    "MESH_2hop",
    "GDA_2hop",
    # 10~13：药物相似图和 ADR 相似图共同传播，刻画双侧邻居的一致性。
    "DGen_to_MESH",
    "DGen_to_GDA",
    "CS_to_MESH",
    "CS_to_GDA",
    # 14~15：由关联矩阵自身导出的药物超图和 ADR 超图传播结果。
    "drug_hypergraph",
    "adr_hypergraph",
]


def _positive_visible_matrix(shape, train_data):
    """只将基础训练子集中的真实正边写入可见关联矩阵。"""
    # train_data 的每行格式为 [drug_idx, adr_idx, rating]。rating 大于零表示真实已观察关联；
    # 负样本和 RPU 采样得到的未知负样本都不能写入 A，避免它们被误当成拓扑结构。
    # 返回矩阵 A 的形状为 (药物数, ADR 数)，后续全部拓扑特征均只从该矩阵计算。
    visible = np.zeros(shape, dtype=np.float32)
    samples = np.asarray(train_data)
    positives = samples[samples[:, 2].astype(np.float32) > 0]
    visible[
        positives[:, 0].astype(np.int64),
        positives[:, 1].astype(np.int64),
    ] = 1.0
    return visible


def _normalized_topk(similarity, topk):
    """将方阵相似度转为行归一化的稀疏邻接矩阵。"""
    # 输入和输出均为 (实体数, 实体数)。每一行代表一个实体到同侧实体的传播权重。
    matrix = np.asarray(similarity, dtype=np.float32).copy()
    # 相似度只作为传播权重，异常值、负值和自环均不参与邻居投票。
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(matrix, 0.0, out=matrix)#两者取最大
    np.fill_diagonal(matrix, 0.0)

    neighbor_count = min(int(topk), matrix.shape[0] - 1)
    if neighbor_count <= 0:
        return np.zeros_like(matrix)

    #只保留每行 top-k 个元素（在原本的位置上）
    indices = np.argpartition(matrix, -neighbor_count, axis=1)[:, -neighbor_count:]
    sparse_matrix = np.zeros_like(matrix)
    rows = np.repeat(np.arange(matrix.shape[0]), neighbor_count)
    sparse_matrix[rows, indices.reshape(-1)] = matrix[rows, indices.reshape(-1)]#给空矩阵每行赋值上最大的k个位置元素
    row_sums = sparse_matrix.sum(axis=1, keepdims=True)
    # 没有可用邻居的行保持全零；分母设为 1 仅用于避免除零。
    row_sums[row_sums == 0] = 1.0
    return sparse_matrix / row_sums


def _inverse_degree(values):
    """计算超图传播所需的逆度数，对零度节点保持零贡献。
    
    在超图视角下，每条超边（如一个ADR）连接多个节点（药物）。
    当多个药物共享同一个ADR时，该ADR对药物共现的贡献会被其关联的药物数量均摊。
    度数越大（关联药物越多）的ADR越"泛滥"，其区分度越低，应被抑制。
    
    例如：ADR"头痛"关联了100个药物，ADR"罕见肝损伤"只关联了2个药物。
    两者对药物共现的原始计数贡献相同（都是1），但逆度数归一化后，
    前者权重为1/100=0.01，后者为1/2=0.5，稀有ADR的信号被放大50倍。
    
    零度节点（无任何关联的孤立ADR）返回0而非inf，避免除零错误。
    """
    inverse = np.zeros_like(values, dtype=np.float32)
    nonzero = values > 0
    inverse[nonzero] = 1.0 / values[nonzero]
    return inverse


@dataclass
class TopologyScorer:
    """保存一个外层折内唯一允许使用的拓扑特征。"""

    visible_matrix: np.ndarray
    multiscale_scores: np.ndarray
    drug_degrees: np.ndarray
    side_degrees: np.ndarray

    def pair_features(self, drug_indices, side_indices):
        """按 pair 索引取出 16 个折内拓扑特征，不包含 CNN logit。"""
        # drug_indices 与 side_indices 长度均为 N；第 i 个位置共同确定第 i 个 pair。
        # 返回形状为 (N, 16)，main.py 再把同顺序的 CNN logit 拼到第一列形成 (N, 17)。
        drug_indices = np.asarray(drug_indices, dtype=np.int64)
        side_indices = np.asarray(side_indices, dtype=np.int64)
        degrees = np.column_stack((
            np.log1p(self.drug_degrees[drug_indices]),
            np.log1p(self.side_degrees[side_indices]),
        ))
        return np.column_stack((
            degrees,
            self.multiscale_scores[drug_indices, side_indices],
        )).astype(np.float32)


def build_topology_scorer(
        drug_similarities, side_similarities, train_data, shape,
        drug_topk=40, adr_topk=80):
    """构建一个外层折使用的一跳、二跳、双侧和超图传播特征表。"""
    if len(drug_similarities) != 2 or len(side_similarities) != 2:
        raise ValueError("拓扑校准层需要 DGen、CS、MESH、GDA 四个外部视图。")

    # A 只含基础训练正边；验证、校准和测试边均不写入，保证全部传播 fold-safe。
    # 相似度图不含标签，而每一个与 A 相乘的结果都只依赖当前基础训练折。
    visible = _positive_visible_matrix(shape, train_data)
    drug_graphs = [_normalized_topk(similarity, drug_topk) for similarity in drug_similarities]#保留topk且归一化
    side_graphs = [_normalized_topk(similarity, adr_topk) for similarity in side_similarities]

    # 药物侧为 S_drug @ A，ADR 侧为 A @ S_adr，共四个一跳分数。
    # 前两张矩阵形状均为 (药物数, ADR数)，后两张矩阵形状也相同，能够直接按 pair 取值。
    one_hop = [graph @ visible for graph in drug_graphs]
    one_hop.extend(visible @ graph for graph in side_graphs)

    # 在各自一跳结果上再次传播，得到 S_drug^2 @ A 与 A @ S_adr^2。
    # 仍然只是在折内可见矩阵 A 上扩散，不会把隐藏 pair 的标签带入特征。
    two_hop = [graph @ score for graph, score in zip(drug_graphs, one_hop[:2])]
    two_hop.extend(score @ graph for score, graph in zip(one_hop[2:], side_graphs))

    # 两个药物图与两个 ADR 图两两组合，得到四个 S_drug @ A @ S_adr 分数。
    # 每一项同时要求药物邻域与 ADR 邻域支持同一 pair，因此称为双侧传播。
    dual_side = [drug_graph @ visible @ side_graph
                 for drug_graph in drug_graphs for side_graph in side_graphs]

    drug_degrees = visible.sum(axis=1).astype(np.float32)
    side_degrees = visible.sum(axis=0).astype(np.float32)

    # 两个结果的形状都保持为 (药物数, ADR数)，可与前述 14 个传播特征堆叠。
    #通过drug/adr超边来传递信息
    #_inverse_degree控制信息传播大小,例如drug-drug共现的时候,ADR j 的度数越大（越"泛滥"），它对药物共现的贡献被压得越小。
    drug_hypergraph = (visible * _inverse_degree(side_degrees)) @ visible.T @ visible#visible当特征矩阵的超图传播(从右往左)
    side_hypergraph = visible @ (visible.T * _inverse_degree(drug_degrees)) @ visible

    return TopologyScorer(
        visible_matrix=visible,
        multiscale_scores=np.stack(
            one_hop + two_hop + dual_side + [drug_hypergraph, side_hypergraph], axis=-1
        ).astype(np.float32),
        drug_degrees=drug_degrees,
        side_degrees=side_degrees,
    )


def assert_hidden_pairs_are_masked(visible_matrix, hidden_data):
    """确认验证、校准和测试 pair 均未出现在可见关联矩阵中。"""
    hidden = np.asarray(hidden_data)
    if len(hidden) == 0:
        return
    values = visible_matrix[
        hidden[:, 0].astype(np.int64),
        hidden[:, 1].astype(np.int64),
    ]
    if np.any(values != 0):
        raise RuntimeError("隐藏 pair 出现在拓扑可见矩阵中，终止以避免标签泄露。")
