"""Heterogeneous Attention Network (HAN) Module

Implements HAN for heterogeneous graph learning with:
- Node-level attention: Attention within each meta-path
- Semantic-level attention: Attention across different meta-paths
- Edge embeddings: Learnable embeddings for different edge types
- [NEW] Dynamic Edge Weight Learning: 动态学习边权重
- [NEW] Hierarchical Layer Aggregation: 分层聚合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add  # 用于高效的scatter操作
from torch_geometric.utils import softmax as geo_softmax  # 高效的图softmax
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# [NEW] Dynamic Edge Weight Learning Module - 动态边权重学习
# ============================================================================

class DynamicEdgeWeightLearner(nn.Module):
    """动态边权重学习模块
    
    创新点：基于源节点和目标节点的特征动态计算边权重，
    而不是使用固定的边权重。这允许模型学习不同drug-gene
    或gene-adr关系的重要性差异。
    
    计算公式: w_ij = sigmoid(MLP([h_i || h_j || e_ij]))
    其中 h_i, h_j 是节点特征，e_ij 是边类型嵌入
    """
    
    def __init__(self, node_dim: int, edge_dim: int = 32, hidden_dim: int = 64):
        """初始化动态边权重学习器
        
        Args:
            node_dim: 节点特征维度
            edge_dim: 边嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(DynamicEdgeWeightLearner, self).__init__()
        
        # 输入: [src_feat || dst_feat || edge_emb] -> 边权重
        input_dim = node_dim * 2 + edge_dim
        
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1] 范围的权重
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使初始边权重接近1"""
        for m in self.weight_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 偏置初始化使sigmoid输出接近0.5
                    nn.init.zeros_(m.bias)
    
    def forward(self, src_feat: Tensor, dst_feat: Tensor, 
                edge_emb: Tensor) -> Tensor:
        """计算动态边权重
        
        Args:
            src_feat: 源节点特征 [num_edges, node_dim]
            dst_feat: 目标节点特征 [num_edges, node_dim]
            edge_emb: 边类型嵌入 [num_edges, edge_dim]
            
        Returns:
            edge_weights: 边权重 [num_edges, 1]
        """
        # 拼接特征: [num_edges, node_dim*2 + edge_dim]
        combined = torch.cat([src_feat, dst_feat, edge_emb], dim=-1)
        
        # 计算边权重: [num_edges, 1]
        edge_weights = self.weight_mlp(combined)
        
        return edge_weights


class EdgeEmbedding(nn.Module):
    """Learnable edge embeddings for different edge types.
    
    Edge types:
    0: drug_to_gene (Drug -> Gene, 正向)
    1: gene_to_adr (Gene -> ADR, 正向)
    2: gene_to_drug (Gene -> Drug, 反向，用于adr->gene->drug传播)
    3: adr_to_gene (ADR -> Gene, 反向，用于adr->gene->drug传播)
    """
    
    def __init__(self, num_edge_types: int = 4, edge_dim: int = 32):
        """Initialize edge embedding.
        
        Args:
            num_edge_types: Number of edge types (4: drug_to_gene, gene_to_adr, gene_to_drug, adr_to_gene)
            edge_dim: Dimension of edge embeddings
        """
        super(EdgeEmbedding, self).__init__()
        self.num_edge_types = num_edge_types
        self.edge_dim = edge_dim
        self.embedding = nn.Embedding(num_edge_types, edge_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, edge_types: Tensor) -> Tensor:
        """Get edge embeddings.
        
        Args:
            edge_types: Edge type indices [num_edges] (0-3)
            
        Returns:
            Edge embeddings [num_edges, edge_dim]
        """
        return self.embedding(edge_types)


class NodeLevelAttention(nn.Module):
    """Node-level attention mechanism with Multi-Scale support.
    
    Integrates multi-scale attention to capture:
    - Local patterns (1-hop neighbors)
    - Medium-range patterns (2-hop neighbors)
    
    [NEW] 支持动态边权重学习
    Note: Global attention has been removed to improve performance.
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 1, 
                 dropout: float = 0.1, edge_dim: int = 32, use_edge_embedding: bool = True,
                 use_multi_scale: bool = False, scales: List[str] = ['local'],
                 use_dynamic_edge_weight: bool = False):
        """Initialize node-level attention.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_dim: Edge embedding dimension
            use_edge_embedding: Whether to use edge embeddings
            use_multi_scale: Whether to use multi-scale attention
            scales: List of scales to use ['local', 'medium']
            use_dynamic_edge_weight: [NEW] 是否使用动态边权重学习
        """
        super(NodeLevelAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_embedding = use_edge_embedding
        self.edge_dim = edge_dim if use_edge_embedding else 0
        self.use_multi_scale = use_multi_scale
        self.scales = scales if use_multi_scale else ['local']
        self.use_dynamic_edge_weight = use_dynamic_edge_weight
        
        # Linear transformation for each head
        self.W = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))#初始化为内存未定义随机值后面reset_parameters()会重新初始化
        
        # Attention weight computation: [h_src, h_dst, edge_emb] -> attention score
        attn_input_dim = 2 * out_dim + self.edge_dim
        self.a = nn.Parameter(torch.empty(num_heads, attn_input_dim, 1))
        
        # Multi-scale fusion (if enabled)
        if self.use_multi_scale and len(self.scales) > 1:
            # 每个scale产生num_heads * out_dim维特征
            total_scale_dim = len(self.scales) * num_heads * out_dim
            self.scale_fusion = nn.Linear(total_scale_dim, num_heads * out_dim)#[1024,512]
            self.scale_layer_norm = nn.LayerNorm(num_heads * out_dim)#[512,]
        
        # [NEW] 动态边权重学习模块
        if self.use_dynamic_edge_weight:
            self.edge_weight_learner = DynamicEdgeWeightLearner(
                node_dim=in_dim,
                edge_dim=edge_dim,
                hidden_dim=64
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_types: Optional[Tensor] = None,
                edge_embedding: Optional[nn.Module] = None) -> Tensor:
        """Apply node-level attention with optional multi-scale support.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge type indices [num_edges] (0-3), optional
            edge_embedding: EdgeEmbedding module, optional
            
        Returns:
            Output features [num_nodes, num_heads * out_dim]
        """
        if not self.use_multi_scale:#使用多个尺度的节点注意力
            # Standard single-scale attention
            return self._single_scale_attention(x, edge_index, edge_types, edge_embedding)
        else:
            # Multi-scale attention
            return self._multi_scale_attention(x, edge_index, edge_types, edge_embedding)
    
    def _single_scale_attention(self, x: Tensor, edge_index: Tensor, 
                               edge_types: Optional[Tensor] = None,
                               edge_embedding: Optional[nn.Module] = None) -> Tensor:
        """Standard single-scale attention (original implementation).
        
        [MODIFIED] 支持动态边权重学习
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge type indices [num_edges]
            edge_embedding: EdgeEmbedding module
            
        Returns:
            Output features [num_nodes, num_heads * out_dim]
        """
        num_nodes = x.size(0)#节点个数
        num_edges = edge_index.size(1)#节点的边个数
        
        if num_edges == 0:
            # No edges, return zero features
            return torch.zeros(num_nodes, self.num_heads * self.out_dim, device=x.device)
        
        # x: [num_nodes, in_dim] 广播为 [1, num_nodes, in_dim]
        # self.W: [num_heads, in_dim, out_dim]
        # 结果: [num_heads, num_nodes, out_dim]
        # Transform features: [num_nodes, in_dim] -> [num_heads, num_nodes, out_dim]
        h = torch.matmul(x, self.W)  # [num_heads, num_nodes, out_dim]
        h = h.transpose(0, 1).contiguous()  # [num_nodes, num_heads, out_dim]

        # Compute attention scores for each edge
        src, dst = edge_index[0], edge_index[1]
        h_src = h[src]  #  [num_edges, num_heads, out_dim]#取源节点处理过的特征
        h_dst = h[dst]  #  [num_edges, num_heads, out_dim]#取目标节点处理过的特征
        
        # Concatenate node features
        h_concat = torch.cat([h_src, h_dst], dim=-1)  # [num_edges, num_heads, 2*out_dim]
        
        # Add edge embeddings if available
        edge_emb = None
        if self.use_edge_embedding and edge_types is not None and edge_embedding is not None:
            edge_emb = edge_embedding(edge_types)  # [num_edges, edge_dim]
            # Expand edge_emb to match num_heads: [num_edges, edge_dim] -> [num_edges, num_heads, edge_dim]
            edge_emb_expanded = edge_emb.unsqueeze(1).expand(-1, self.num_heads, -1)
            # Concatenate: [num_edges, num_heads, 2*out_dim + edge_dim]
            h_concat = torch.cat([h_concat, edge_emb_expanded], dim=-1)#把对应的边的特征直接最后一个维度concat
        
        # Compute attention scores
        # h_concat: [num_edges, num_heads, attn_input_dim]
        # self.a: [num_heads, attn_input_dim, 1]
        # 需要对每个 head 分别计算注意力分数
        # 方法：重排维度使 num_heads 成为批次维度
        h_concat_transposed = h_concat.transpose(0, 1)  # [num_heads, num_edges, attn_input_dim]
        e = torch.matmul(h_concat_transposed, self.a).squeeze(-1)  # [num_heads, num_edges]  
        e = e.transpose(0, 1)  # [num_edges, num_heads]
        e = F.leaky_relu(e, negative_slope=0.2)
        
        # 使用 PyG 的高效 softmax（向量化处理所有heads）
        # 将 [num_edges, num_heads] 展平为 [num_edges * num_heads]
        # 然后用扩展的 dst 索引进行 softmax
        num_edges_actual = e.size(0)
        e_flat = e.transpose(0, 1).reshape(-1)  # [num_heads * num_edges]
        dst_expanded = dst.repeat(self.num_heads) + torch.arange(self.num_heads, device=x.device).repeat_interleave(num_edges_actual) * num_nodes
        alpha_flat = geo_softmax(e_flat, dst_expanded, num_nodes=num_nodes * self.num_heads)
        alpha = alpha_flat.view(self.num_heads, num_edges_actual).transpose(0, 1)  # [num_edges, num_heads]
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # [NEW] 动态边权重学习 - 在注意力权重基础上乘以学习到的边权重
        if self.use_dynamic_edge_weight and edge_emb is not None:
            # 获取原始节点特征用于边权重计算
            src_feat = x[src]  # [num_edges, in_dim]
            dst_feat = x[dst]  # [num_edges, in_dim]
            
            # 计算动态边权重: [num_edges, 1]
            dynamic_weights = self.edge_weight_learner(src_feat, dst_feat, edge_emb)
            
            # 将动态权重应用到注意力分数上: alpha * dynamic_weight
            # dynamic_weights: [num_edges, 1] -> [num_edges, num_heads]
            alpha = alpha * dynamic_weights  # Broadcasting: [num_edges, num_heads] * [num_edges, 1]
        
        # Weight neighbor features
        h_weighted = h_src * alpha.unsqueeze(-1)  # [num_edges, num_heads, out_dim]用注意力分数与源节点特征元素级乘法
        
        # Aggregate: 使用高效的scatter_add操作替代循环
        # 首先初始化输出张量为节点自身特征（处理没有邻居的节点）
        h_out = h.clone()  # [num_nodes, num_heads, out_dim]
        
        # 使用scatter_add进行高效聚合
        # 将h_weighted按dst节点索引聚合
        h_weighted_flat = h_weighted.view(num_edges, -1)  # [num_edges, num_heads * out_dim]
        h_out_flat = torch.zeros(num_nodes, self.num_heads * self.out_dim, device=x.device)
        
        # scatter_add: 将相同dst节点的特征相加
        scatter_add(h_weighted_flat, dst, dim=0, out=h_out_flat)
        
        # 将有邻居的节点更新为聚合后的特征
        h_out_reshaped = h_out_flat.view(num_nodes, self.num_heads, self.out_dim)
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        mask[dst.unique()] = True
        h_out[mask] = h_out_reshaped[mask]
        
        # Reshape: [num_nodes, num_heads, out_dim] -> [num_nodes, num_heads * out_dim]
        h_out = h_out.view(num_nodes, -1)
        
        return h_out
    
    def _multi_scale_attention(self, x: Tensor, edge_index: Tensor,
                              edge_types: Optional[Tensor] = None,
                              edge_embedding: Optional[nn.Module] = None) -> Tensor:
        """Multi-scale attention combining local, medium, and global features.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge type indices [num_edges]
            edge_embedding: EdgeEmbedding module
            
        Returns:
            Fused multi-scale features [num_nodes, num_heads * out_dim]
        """
        num_nodes = x.size(0)#节点个数
        scale_features = []
        
        for scale in self.scales:#依次遍历设置的尺度进行节点注意力
            if scale == 'local':
                # Local: use original 1-hop edges
                scale_out = self._single_scale_attention(x, edge_index, edge_types, edge_embedding)
            
            elif scale == 'medium':
                # Medium: compute 2-hop neighbors
                edge_index_2hop = self._compute_2hop_edges(edge_index, num_nodes)#通过矩阵自相乘得到 i 和 j 之间存在2跳路径的节点对
                # 2-hop边没有边类型，但需要保持维度一致，创建零边嵌入
                if self.use_edge_embedding and edge_embedding is not None and edge_index_2hop.size(1) > 0:
                    # 创建虚拟边类型（全零）和对应的零嵌入
                    dummy_edge_types = torch.zeros(edge_index_2hop.size(1), dtype=torch.long, device=x.device)
                    scale_out = self._single_scale_attention(x, edge_index_2hop, dummy_edge_types, edge_embedding)
                else:
                    scale_out = self._single_scale_attention(x, edge_index_2hop, None, None)
            
            else:
                # Fallback to local
                scale_out = self._single_scale_attention(x, edge_index, edge_types, edge_embedding)
            
            scale_features.append(scale_out)
        
        # Fuse multi-scale features
        if len(scale_features) == 1:
            return scale_features[0]
        
        fused = torch.cat(scale_features, dim=-1)  # [num_nodes, num_scales * num_heads * out_dim]
        fused = self.scale_fusion(fused)  # [num_nodes, num_heads * out_dim]
        fused = self.scale_layer_norm(fused)
        
        return fused
    
    def _compute_2hop_edges(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """Compute 2-hop neighbors from 1-hop edges.
        
        Args:
            edge_index: 1-hop edge indices [2, num_edges]
            num_nodes: Number of nodes
            
        Returns:
            2-hop edge indices [2, num_2hop_edges]
        """
        if edge_index.size(1) == 0:
            return edge_index
        
        # Build adjacency matrix (sparse)
        src, dst = edge_index[0], edge_index[1]
        device = edge_index.device
        
        # Create sparse adjacency matrix
        adj = torch.sparse_coo_tensor(
            edge_index, 
            torch.ones(edge_index.size(1), device=device),
            (num_nodes, num_nodes)
        )
        
        # Compute A^2 (2-hop adjacency) - 使用纯稀疏矩阵乘法优化
        # 注意：稀疏矩阵乘法结果仍是稀疏矩阵，但CUDA稀疏张量不支持nonzero()
        adj_2hop = torch.sparse.mm(adj, adj)  # A²[i][j] 表示从节点 i 到节点 j 的长度为2的路径数量
        
        # Convert back to edge_index format
        # 方法1: coalesce()整理稀疏矩阵，然后使用indices()
        adj_2hop = adj_2hop.coalesce()  # 合并重复索引
        edge_index_2hop = adj_2hop.indices()  # 直接获取非零元素的索引 [2, num_2hop_edges]
        
        return edge_index_2hop
    
    # _global_attention method removed to improve performance
    # Global attention was causing O(n²) complexity which significantly slowed down training


class SemanticLevelAttention(nn.Module):
    """语义级注意力：融合gene节点从不同入边类型接收的信息。
    
    只有gene节点需要语义融合（接收来自drug和adr两种信息）。
    drug节点只接收gene的信息，adr节点只接收gene的信息。
    """
    
    def __init__(self, in_dim: int, num_edge_types: int = 2, dropout: float = 0.1):
        super(SemanticLevelAttention, self).__init__()
        self.in_dim = in_dim
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        
        # 每种入边类型一组独立参数
        self.q_list = nn.ParameterList([
            nn.Parameter(torch.empty(1, in_dim)) for _ in range(num_edge_types)
        ])
        self.W_list = nn.ModuleList([
            nn.Linear(in_dim, in_dim) for _ in range(num_edge_types)
        ])
        self.reset_parameters()
    
    def reset_parameters(self):
        for q in self.q_list:
            nn.init.xavier_uniform_(q)
        for W in self.W_list:
            nn.init.xavier_uniform_(W.weight)
            nn.init.zeros_(W.bias)
    
    def forward(self, z_list: List[Tensor]) -> Tensor:
        """对gene节点融合多种入边信息。
        
        Args:
            z_list: 来自不同入边类型的特征列表，每个[num_gene_nodes, in_dim]
        Returns:
            融合后的gene节点特征 [num_gene_nodes, in_dim]
        """
        if len(z_list) == 1:
            return z_list[0]
        
        z_stack = torch.stack(z_list, dim=0)  # [num_types, num_genes, dim]
        
        # 计算每种入边类型的注意力分数
        scores = []
        for i, z in enumerate(z_list):
            transformed = self.W_list[i](z)
            score = torch.matmul(transformed, self.q_list[i].t()).squeeze(-1)
            scores.append(score)
        
        scores = torch.stack(scores, dim=0)  # [num_types, num_genes]
        scores = F.leaky_relu(scores, negative_slope=0.2)
        beta = F.softmax(scores, dim=0)
        beta = F.dropout(beta, p=self.dropout, training=self.training)
        
        # 加权融合
        z_fused = (z_stack * beta.unsqueeze(-1)).sum(dim=0)
        return z_fused


class HANLayer(nn.Module):
    """异构图注意力层：按入边类型分别聚合，只对gene节点做语义融合。
    
    [NEW] 支持动态边权重学习
    
    边类型定义:
    - 0: drug->gene  (gene接收drug信息)
    - 1: gene->adr   (adr接收gene信息)
    - 2: gene->drug  (drug接收gene信息)
    - 3: adr->gene   (gene接收adr信息)
    
    消息传递规则:
    - drug节点: 只从gene->drug边接收信息
    - adr节点: 只从gene->adr边接收信息
    - gene节点: 从drug->gene和adr->gene边接收信息，需要语义融合
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, num_metapaths: int = 2, edge_dim: int = 32,
                 use_edge_embedding: bool = True, use_multi_scale: bool = False,
                 scales: List[str] = ['local'], use_dynamic_edge_weight: bool = False):
        super(HANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_embedding = use_edge_embedding
        self.use_dynamic_edge_weight = use_dynamic_edge_weight
        
        # 4种边类型各自独立的节点级注意力
        # 0: drug->gene, 1: gene->adr, 2: gene->drug, 3: adr->gene
        self.node_attentions = nn.ModuleList([
            NodeLevelAttention(in_dim, out_dim, num_heads, dropout, edge_dim,
                             use_edge_embedding, use_multi_scale, scales,
                             use_dynamic_edge_weight)  # [NEW] 传递动态边权重参数
            for _ in range(4)
        ])
        
        # gene节点的语义融合（融合drug->gene和adr->gene两种入边）
        self.semantic_attention = SemanticLevelAttention(
            num_heads * out_dim, num_edge_types=2, dropout=dropout
        )
        
        # 输出投影
        self.out_proj = nn.Linear(num_heads * out_dim, out_dim)
    
    def forward(self, x: Tensor, edge_dict: Dict[int, Tensor],
                edge_embedding: Optional[nn.Module] = None,
                node_types: Optional[Tensor] = None) -> Tensor:
        """按入边类型分别聚合，然后按节点类型组装输出。
        
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_dict: 边类型->边索引的字典 {edge_type: [2, num_edges]}
            edge_embedding: 边嵌入模块
            node_types: 节点类型 [num_nodes] (0=drug, 1=gene, 2=adr)
        Returns:
            输出特征 [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        device = x.device
        hidden_dim = self.num_heads * self.out_dim
        
        # 对每种边类型分别做消息传递
        z_by_edge_type = {}
        for edge_type in range(4):
            edge_index = edge_dict.get(edge_type, torch.empty((2, 0), dtype=torch.long, device=device))
            if edge_index.size(1) > 0:
                edge_types_tensor = torch.full((edge_index.size(1),), edge_type, dtype=torch.long, device=device)
                z_by_edge_type[edge_type] = self.node_attentions[edge_type](
                    x, edge_index, edge_types_tensor, edge_embedding
                )
            else:
                z_by_edge_type[edge_type] = torch.zeros(num_nodes, hidden_dim, device=device)
        
        # 初始化输出（使用输入特征的投影作为默认值）
        z_output = torch.zeros(num_nodes, hidden_dim, device=device)
        
        if node_types is not None:
            # drug节点(type=0): 取gene->drug边(type=2)的聚合结果
            drug_mask = (node_types == 0)
            if drug_mask.any():
                z_output[drug_mask] = z_by_edge_type[2][drug_mask]
            
            # adr节点(type=2): 取gene->adr边(type=1)的聚合结果
            adr_mask = (node_types == 2)
            if adr_mask.any():
                z_output[adr_mask] = z_by_edge_type[1][adr_mask]
            
            # gene节点(type=1): 融合drug->gene(type=0)和adr->gene(type=3)
            gene_mask = (node_types == 1)
            if gene_mask.any():
                z_from_drug = z_by_edge_type[0][gene_mask]  # [num_genes, hidden_dim]
                z_from_adr = z_by_edge_type[3][gene_mask]   # [num_genes, hidden_dim]
                z_gene_fused = self.semantic_attention([z_from_drug, z_from_adr])
                z_output[gene_mask] = z_gene_fused
        else:
            # 无节点类型信息时，简单取第一个边类型结果
            z_output = z_by_edge_type[0]
        
        # 输出投影
        return self.out_proj(z_output)


class HANEncoder(nn.Module):
    """多层异构图注意力编码器。
    
    [NEW] 支持分层聚合 (Hierarchical Layer Aggregation)
    [NEW] 支持动态边权重学习
    
    边类型: 0=drug->gene, 1=gene->adr, 2=gene->drug, 3=adr->gene
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 num_metapaths: int = 2, use_residual: bool = True,
                 edge_dim: int = 32, use_edge_embedding: bool = True,
                 use_multi_scale: bool = False, scales: List[str] = ['local'],
                 use_hierarchical_agg: bool = False,
                 use_dynamic_edge_weight: bool = False):
        """初始化HAN编码器
        
        Args:
            in_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            out_dim: 输出特征维度
            num_layers: HAN层数
            num_heads: 注意力头数
            dropout: Dropout率
            num_metapaths: 元路径数量
            use_residual: 是否使用残差连接
            edge_dim: 边嵌入维度
            use_edge_embedding: 是否使用边嵌入
            use_multi_scale: 是否使用多尺度注意力
            scales: 尺度列表
            use_hierarchical_agg: [NEW] 是否使用分层聚合
            use_dynamic_edge_weight: [NEW] 是否使用动态边权重
        """
        super(HANEncoder, self).__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_hierarchical_agg = use_hierarchical_agg
        self.use_dynamic_edge_weight = use_dynamic_edge_weight
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            HANLayer(hidden_dim, hidden_dim, num_heads, dropout,
                    num_metapaths, edge_dim, use_edge_embedding,
                    use_multi_scale, scales, use_dynamic_edge_weight)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # ================================================================
        # [NEW] 分层聚合模块 (Hierarchical Layer Aggregation)
        # ================================================================
        if self.use_hierarchical_agg:
            # 可学习的层权重: 每层一个权重，用于加权聚合各层输出
            # 初始化为均匀分布，让模型自己学习最优权重
            self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))  # +1 for input layer
            
            # 层级注意力: 基于层输出特征动态计算层权重
            self.layer_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # 最终融合层: 将加权聚合的特征投影到输出维度
            self.hierarchical_fusion = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: Tensor, edge_dict: Dict[int, Tensor],
                edge_embedding: Optional[nn.Module] = None,
                node_types: Optional[Tensor] = None) -> Tensor:
        """多层异构图注意力编码。
        
        [MODIFIED] 支持分层聚合
        
        Args:
            x: 节点特征 [num_nodes, in_dim]
            edge_dict: 边类型->边索引字典 {0: drug->gene, 1: gene->adr, 2: gene->drug, 3: adr->gene}
            edge_embedding: 边嵌入模块
            node_types: 节点类型 [num_nodes] (0=drug, 1=gene, 2=adr)
        Returns:
            输出特征 [num_nodes, hidden_dim]
        """
        # 输入投影
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout(h)
        
        # [NEW] 收集各层输出用于分层聚合
        if self.use_hierarchical_agg:
            layer_outputs = [h]  # 包含输入层
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_dict, edge_embedding, node_types)
            
            # 残差连接
            if self.use_residual and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            h = self.layer_norms[i](h)
            
            # [NEW] 收集当前层输出
            if self.use_hierarchical_agg:
                layer_outputs.append(h)
            
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        # ================================================================
        # [NEW] 分层聚合: 加权融合各层输出
        # ================================================================
        if self.use_hierarchical_agg:
            h = self._hierarchical_aggregate(layer_outputs)
        
        return h
    
    def _hierarchical_aggregate(self, layer_outputs: List[Tensor]) -> Tensor:
        """分层聚合各层输出
        
        创新点：
        1. 使用可学习的层权重，让模型自动学习各层的重要性
        2. 使用层级注意力机制，基于节点特征动态调整层权重
        3. 浅层保留局部结构信息，深层捕获全局语义信息
        
        Args:
            layer_outputs: 各层输出列表 [h_0, h_1, ..., h_L]
                          每个 h_i: [num_nodes, hidden_dim]
        
        Returns:
            aggregated: 聚合后的特征 [num_nodes, hidden_dim]
        """
        num_layers = len(layer_outputs)
        num_nodes = layer_outputs[0].size(0)
        device = layer_outputs[0].device
        
        # 方法1: 基于可学习权重的静态聚合
        # 对层权重做softmax归一化
        normalized_weights = F.softmax(self.layer_weights[:num_layers], dim=0)
        
        # 加权求和: sum(w_i * h_i)
        aggregated = torch.zeros_like(layer_outputs[0])
        for i, h in enumerate(layer_outputs):
            aggregated = aggregated + normalized_weights[i] * h
        
        # 方法2: 基于注意力的动态聚合 (与方法1结合)
        # 计算每层每个节点的注意力分数
        layer_stack = torch.stack(layer_outputs, dim=0)  # [num_layers, num_nodes, hidden_dim]
        
        # 计算层级注意力分数
        attn_scores = self.layer_attention(layer_stack)  # [num_layers, num_nodes, 1]
        attn_scores = attn_scores.squeeze(-1).transpose(0, 1)  # [num_nodes, num_layers]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [num_nodes, num_layers]
        
        # 动态加权聚合
        layer_stack_transposed = layer_stack.transpose(0, 1)  # [num_nodes, num_layers, hidden_dim]
        dynamic_aggregated = torch.bmm(
            attn_weights.unsqueeze(1),  # [num_nodes, 1, num_layers]
            layer_stack_transposed      # [num_nodes, num_layers, hidden_dim]
        ).squeeze(1)  # [num_nodes, hidden_dim]
        
        # 融合静态和动态聚合结果 (各占50%)
        final_aggregated = 0.5 * aggregated + 0.5 * dynamic_aggregated
        
        # 最终投影
        final_aggregated = self.hierarchical_fusion(final_aggregated)
        
        return final_aggregated

