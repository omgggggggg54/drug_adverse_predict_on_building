"""DGAPred Model Architecture

This module implements the DGAPred model with advanced components:
- Bipartite Graph Attention Networks (BiMPADR style)
- Dual-Attention Graph Transformer (DrugDAGT 2024)
- Cross-Modal Attention (CM-DTA 2025)
- Contrastive Learning (CCL-ASPS 2024)
"""

import math
import numpy as np
from typing import Optional, Union, Tuple

from sympy.printing.c import none
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor, Adj, Size

from .resnet import Residual
from .ARConv import ARConv
from .han import HANEncoder, EdgeEmbedding
from utils.heterogeneous_graph import HeterogeneousGraphBuilder, KHopSubgraphSampler

# ============================================================================
# Utility Functions
# ============================================================================

def glorot(tensor: Optional[Tensor]) -> None:
    """Glorot/Xavier uniform initialization for neural network weights."""
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor: Optional[Tensor]) -> None:
    """Initialize tensor with zeros."""
    if tensor is not None:
        tensor.data.fill_(0)


# ============================================================================
# Contrastive Learning Module
# ============================================================================

class ContrastiveLearningModule(nn.Module):
    """Collaborative Contrastive Learning Module (CCL-ASPS 2024)
    
    Implements contrastive learning between individual views and fused features
    to enhance representation quality.
    """
    def __init__(self, feature_dim: int, temperature: float = 0.07):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
    
    def forward(self, features_list: list, fused_features: torch.Tensor) -> torch.Tensor:
        """Compute collaborative contrastive loss.
        
        Args:
            features_list: List of features from different views
            fused_features: Fused representation(原装)
            
        Returns:
            Average contrastive loss across all views
        """
        fused_proj = self.projection_head(fused_features)
        fused_proj = F.normalize(fused_proj, dim=-1)
        
        total_loss = 0.0
        for view_features in features_list:
            view_proj = self.projection_head(view_features)
            view_proj = F.normalize(view_proj, dim=-1)
            
            # Compute similarity matrix
            similarity = torch.matmul(view_proj, fused_proj.T) / self.temperature
            
            # Positive pairs are on the diagonal
            batch_size = similarity.size(0)
            labels = torch.arange(batch_size, device=similarity.device)
            
            # InfoNCE loss
            loss = F.cross_entropy(similarity, labels)
            total_loss += loss
        
        return total_loss / len(features_list)


# ============================================================================
# Feature Interaction Attention Module
# ============================================================================

class FeatureInteractionAttention(nn.Module):
    """Feature Interaction Attention using Feature Chunks as Sequences (FIA-DTA 2025)
    
    Uses drug and side effect feature chunks as sequence elements for real attention.
    This allows the model to learn which drug features (DGen/GE/CS/Morgan) are most
    relevant to which side effect features (MESH/GDA).
    """
    
    def __init__(self, chunk_dim: int, num_heads: int = 4):
        """Initialize feature interaction attention.
        
        Args:
            chunk_dim: Dimension of each chunk (embed_dim)
            num_heads: Number of attention heads
        """
        super(FeatureInteractionAttention, self).__init__()
        self.num_heads = num_heads
        self.chunk_dim = chunk_dim
        
        # Ensure chunk_dim is divisible by num_heads
        assert chunk_dim % num_heads == 0, f"chunk_dim ({chunk_dim}) must be divisible by num_heads ({num_heads})"
        
        # Drug chunks attend to side chunks
        self.drug_to_side_attn = nn.MultiheadAttention(
            embed_dim=chunk_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Side chunks attend to drug chunks
        self.side_to_drug_attn = nn.MultiheadAttention(
            embed_dim=chunk_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm_drug = nn.LayerNorm(chunk_dim)
        self.layer_norm_side = nn.LayerNorm(chunk_dim)
        
    def forward(self, drug_chunks_list: list, side_chunks_list: list):
        """
        Feature interaction attention between drug and side effect chunks.
        
        Args:
            drug_chunks_list: List of drug chunk tensors, each [batch, chunk_dim]
                             e.g., [x_drug1, x_drug2, x_drug3, x_drug4]
            side_chunks_list: List of side chunk tensors, each [batch, chunk_dim]
                             e.g., [x_side1, x_side2]
        
        Returns:
            enhanced_drug_chunks: List of enhanced drug chunks
            enhanced_side_chunks: List of enhanced side chunks
        """
        # Stack chunks as sequences
        drug_seq = torch.stack(drug_chunks_list, dim=1)  
        # [batch, n_drug_chunks, chunk_dim], e.g., [128, 4, 128]
        
        side_seq = torch.stack(side_chunks_list, dim=1)  
        # [batch, n_side_chunks, chunk_dim], e.g., [128, 2, 128]
        
        # Drug chunks attend to side chunks
        # Each drug chunk learns which side chunks are most relevant
        drug_enhanced, drug_attn_weights = self.drug_to_side_attn(
            query=drug_seq,      # [batch, n_drug_chunks, chunk_dim]
            key=side_seq,        # [batch, n_side_chunks, chunk_dim]
            value=side_seq
        )
        # drug_attn_weights: [batch, n_drug_chunks, n_side_chunks]
        # Shows which side chunks each drug chunk attends to
        
        # Side chunks attend to drug chunks
        # Each side chunk learns which drug chunks are most relevant
        side_enhanced, side_attn_weights = self.side_to_drug_attn(
            query=side_seq,      # [batch, n_side_chunks, chunk_dim]
            key=drug_seq,        # [batch, n_drug_chunks, chunk_dim]
            value=drug_seq
        )
        # side_attn_weights: [batch, n_side_chunks, n_drug_chunks]
        # Shows which drug chunks each side chunk attends to
        
        # Residual connection and layer normalization
        drug_enhanced = self.layer_norm_drug(drug_seq + drug_enhanced)
        side_enhanced = self.layer_norm_side(side_seq + side_enhanced)
        
        # Unstack back to list of chunks
        drug_chunks_enhanced = [drug_enhanced[:, i, :] for i in range(drug_enhanced.size(1))]
        side_chunks_enhanced = [side_enhanced[:, i, :] for i in range(side_enhanced.size(1))]
        
        return drug_chunks_enhanced, side_chunks_enhanced


# ============================================================================
# Main DGAPred Model
# ============================================================================

class DGAPred(nn.Module):
    """Drug-Gene-ADR Prediction Model with Advanced Graph Neural Networks.
    
    Features:
    - Multi-view feature chunking for drugs and side effects
    - Dual-Attention Graph Transformer (DrugDAGT 2024)
    - Feature Interaction Attention (FIA-DTA 2025)
    - Contrastive Learning (CCL-ASPS 2024)
    - ResNet-based interaction map processing
    """
    
    def __init__(self, drugs_dim: int, sides_dim: int, embed_dim: int=128, 
                 batchsize: int=128, dropout1: float = 0.5, dropout2: float = 0.5,
                 n_drug_chunks: int = 2, n_side_chunks: int = 2,
                 use_feature_interaction: bool = True,
                 use_contrastive_learning: bool = True,
                 use_heterogeneous_gnn: bool = False,
                 graph_builder: Optional[HeterogeneousGraphBuilder] = None,
                 han_layers: int = 2, han_heads: int = 4, k_hop: int = 2,
                 max_subgraph_nodes: int = 500, 
                 edge_embed_dim: int = 32, max_gene_neighbors: int = 50,
                 use_adaptive_sampling: bool = True,
                 use_multi_scale:bool=True,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize DGAPred model.
        
        Args:
            drugs_dim: Input dimension for drug features
            sides_dim: Input dimension for side effect features
            embed_dim: Embedding dimension
            batchsize: Batch size
            dropout1: HAN encoder dropout rate
            dropout2: Final prediction dropout rate
            n_drug_chunks: Number of chunks for drug features
            n_side_chunks: Number of chunks for side effect features
            use_feature_interaction: Whether to use feature interaction attention
            use_contrastive_learning: Whether to use contrastive learning
            use_multi_scale: Whether to use multi-scale attention
        """
        super(DGAPred, self).__init__()
        
        # Basic dimensions
        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.embed_dim = embed_dim
        self.batchsize = batchsize
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        # Feature chunking configuration
        self.drug_chunks = n_drug_chunks
        self.side_chunks = n_side_chunks
        self.drug_dim = drugs_dim // self.drug_chunks
        self.side_dim = sides_dim // self.side_chunks
        
        # Advanced technique flags
        self.use_feature_interaction = use_feature_interaction
        self.use_contrastive_learning = use_contrastive_learning
        self.use_heterogeneous_gnn = use_heterogeneous_gnn
        self.use_multi_scale = use_multi_scale
        
        # Heterogeneous GNN configuration
        if self.use_heterogeneous_gnn:
            self.graph_builder = graph_builder
            self.k_hop = k_hop
            self.max_subgraph_nodes = max_subgraph_nodes
            self.edge_embed_dim = edge_embed_dim
            
            # Global learnable embedding for genes (persist across batches/epochs)
            if graph_builder is not None:
                self.num_genes = len(graph_builder.gene_id_to_idx)
                if self.num_genes > 0:
                    self.gene_embedding = nn.Embedding(self.num_genes, embed_dim)
                    nn.init.xavier_uniform_(self.gene_embedding.weight)
                else:
                    self.gene_embedding = None
            else:
                self.num_genes = 0
                self.gene_embedding = None
            #计算各节点的度来得出节点重要性(归一化到[0-1])
            if graph_builder is not None:
                self.subgraph_sampler = KHopSubgraphSampler(
                    graph_builder, k_hop=k_hop, max_nodes=max_subgraph_nodes,
                    max_neighbors_per_node=max_gene_neighbors,  # 限制每个节点的最大邻居数
                    use_adaptive_sampling=use_adaptive_sampling,  # 启用自适应邻居采样
                    use_hierarchical_sampling=True,  # [NEW] 启用分层采样
                    layer_sample_rates=[0.9, 0.5,0.2]  # [NEW] 第1层80%, 第2层40%
                )
            else:
                self.subgraph_sampler = None
            
            # Edge embedding module (4种边类型: drug_to_gene, gene_to_adr, gene_to_drug, adr_to_gene)
            self.edge_embedding = EdgeEmbedding(
                num_edge_types=4,
                edge_dim=edge_embed_dim
            )
            
            # HAN encoder (支持双向传播，每个方向2个meta-path: drug->gene, gene->adr)
            # 支持Multi-Scale Attention: local(1-hop), medium(2-hop)
            # [NEW] 支持分层聚合 (Hierarchical Layer Aggregation)
            # [NEW] 支持动态边权重学习 (Dynamic Edge Weight Learning)
            # 注意: 已移除global注意力以提高训练速度(O(n²)复杂度太高)
            use_multi_scale = self.use_multi_scale  # 默认关闭，可通过参数启用
            scales = ['local']  # 默认只用local scale
            if use_multi_scale:
                scales = ['local', 'medium']  # 启用multi-scale时使用local+medium
            
            self.han_encoder = HANEncoder(
                in_dim=embed_dim,
                hidden_dim=embed_dim,
                out_dim=embed_dim,
                num_layers=han_layers,
                num_heads=han_heads,#HAN编码器层数
                dropout=dropout1,
                num_metapaths=2,  # 每个方向2个meta-path: drug->gene, gene->adr (或adr->gene, gene->drug)
                use_residual=True,
                edge_dim=edge_embed_dim,
                use_edge_embedding=True,
                use_multi_scale=use_multi_scale,
                scales=scales,
                use_hierarchical_agg=True,  # [NEW] 启用分层聚合
                use_dynamic_edge_weight=True  # [NEW] 启用动态边权 重学习
            )
            
            # Feature fusion layer (HAN output + original features)
            self.han_fusion = nn.Linear(embed_dim * 2, embed_dim)
        else:
            self.graph_builder = None
            self.subgraph_sampler = None
            self.han_encoder = None
            self.edge_embedding = None
            self.han_fusion = None
            self.gene_embedding = None
            self.num_genes = 0
        
        # ----------------------------------------------------------------
        # Global feature encoding layers
        # ----------------------------------------------------------------
        self.drugs_layer = nn.Linear(drugs_dim, embed_dim)
        self.drugs_layer_1 = nn.Linear(embed_dim, embed_dim)
        self.drugs_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        self.sides_layer = nn.Linear(sides_dim, embed_dim)
        self.sides_layer_1 = nn.Linear(embed_dim, embed_dim)
        self.sides_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        # Projection back to original dimensions for chunking
        self.drug_back_proj = nn.Linear(embed_dim, drugs_dim)
        self.side_embed_dim = embed_dim  # side embedding 要走图的drug->side传播涉及concat特殊处理
        
        # ----------------------------------------------------------------
        # Chunk-specific feature encoding layers
        # ----------------------------------------------------------------

        # Drug chunk encoders
        self.drug_layer1 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer1_1 = nn.Linear(embed_dim, embed_dim)
        self.drug_layer2 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer2_1 = nn.Linear(embed_dim, embed_dim)
        self.drug_layer3_4 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer3_4_1 = nn.Linear(embed_dim, embed_dim)
        
        self.drug1_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.drug2_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.drug3_4_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        # Side effect chunk encoders
        self.side_layer1 = nn.Linear(self.side_dim, embed_dim)
        self.side_layer1_1 = nn.Linear(embed_dim, embed_dim)
        self.side_layer2 = nn.Linear(self.side_dim, embed_dim)
        self.side_layer2_1 = nn.Linear(embed_dim, embed_dim)
        self.side_layer3 = nn.Linear(self.side_dim, embed_dim)
        self.side_layer3_1 = nn.Linear(embed_dim, embed_dim)
        
        self.side1_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.side2_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.side3_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        # ----------------------------------------------------------------
        # Interaction map processing
        # ----------------------------------------------------------------

        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        # ChemProp 已移除，直接计算 number_map
        self.number_map = self.drug_chunks * self.side_chunks
        
        # 预降采样层：减少显存占用
        # 将 interaction_map 从 [batch, n_maps, 128, 128] 降到 [batch, n_maps, 64, 64]
        # 可减少约 75% 的 ARConv 显存占用，避免 OOM
        self.pre_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ARConv 替换 ResNet: 自适应感受野卷积处理交互图
        self.arconv_interaction = ARConv(
            inc=self.number_map,      # 输入通道数 = drug_chunks * side_chunks
            outc=self.channel_size,   # 输出通道数 = 32
            kernel_size=3,
            padding=1,
            stride=1,
            l_max=9,
            w_max=9,
            flag=False,
            modulation=True
        )
        # 自适应池化：将 [batch, 32, 64, 64] -> [batch, 32, 2, 2]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Finalize side projection layer
        self.side_back_proj = nn.Linear(self.side_embed_dim, self.sides_dim)
        
        # ----------------------------------------------------------------
        # Advanced modules
        # ----------------------------------------------------------------
        
        # Feature Interaction Attention (FIA-DTA 2025)
        if self.use_feature_interaction:
            self.feature_interaction_attn = FeatureInteractionAttention(
                chunk_dim=self.embed_dim,
                num_heads=4
            )
        
        # Contrastive Learning Module (CCL-ASPS 2024)
        if self.use_contrastive_learning:
            # Calculate the correct fused feature dimension
            fused_feature_dim = self.channel_size * 4 + embed_dim + self.side_embed_dim
            self.contrastive_module = ContrastiveLearningModule(
                feature_dim=fused_feature_dim,
                temperature=0.07
            )
        
        # ChemProp 编码器已移除
            
        # ----------------------------------------------------------------
        # Final prediction layers
        # ----------------------------------------------------------------
        
        total_input_dim = self.channel_size * 4 + embed_dim + self.side_embed_dim
        self.total_layer = nn.Linear(total_input_dim, self.channel_size * 4)
        self.total_bn = nn.BatchNorm1d(total_input_dim, momentum=0.5)
        self.classifier = nn.Linear(self.channel_size * 4, 1)
        self.classifier2 = nn.Linear(self.channel_size * 4, 1)  # Outputs logits
        self.con_layer = nn.Linear(self.channel_size * 4, 1)
    
    def make_layer(self, block: nn.Module, channels: int, num_blocks: int, 
                   k_size: int, stride: int) -> nn.Sequential:
        """Create a sequence of residual blocks.
        
        Args:
            block: Residual block class
            channels: Output channels
            num_blocks: Number of blocks to stack
            k_size: Kernel size
            stride: Stride size
            
        Returns:
            Sequential container of residual blocks
        """
        inchannel = self.number_map
        layers = []
        for i in range(num_blocks):
            layers.append(block(inchannel, channels, k_size, stride))
            inchannel = channels
        return nn.Sequential(*layers)


    def forward(self, drug_indices: torch.Tensor, side_indices: torch.Tensor, 
                device: torch.device, global_drug_features: torch.Tensor, 
                global_side_features: torch.Tensor, epoch: int = 0
               ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of DGAPred model.
        
        Args:
            drug_indices: Global indices of drugs in batch [batch_size]
            side_indices: Global indices of side effects in batch [batch_size]
            device: Computing device (CPU or CUDA)
            global_drug_features: Global drug feature matrix [n_drugs, drug_dim]
            global_side_features: Global side effect feature matrix [n_sides, side_dim]
            epoch: Current training epoch (for ARConv adaptive receptive field)
         
            
        Returns:
            classification: Classification logits [batch_size]
            regression: Regression predictions [batch_size]
            contrastive_loss (optional): Contrastive loss if training
        """
        # ----------------------------------------------------------------
        # Step 1: Move data to device
        # ----------------------------------------------------------------
        global_drug_features = global_drug_features.to(device)
        global_side_features = global_side_features.to(device)
        drug_indices = drug_indices.to(device)
        side_indices = side_indices.to(device)
        
        batch_drug_features = global_drug_features[drug_indices]
        batch_side_features = global_side_features[side_indices]
        
        # ----------------------------------------------------------------
        # Step 2: Global encoding (without BipartiteGAT)
        # ----------------------------------------------------------------
        contrastive_loss = None
        
        # Directly encode batch features
        x_drugs_embed = F.relu(self.drugs_bn(self.drugs_layer(batch_drug_features)), inplace=True)
        x_drugs_embed = F.dropout(x_drugs_embed, training=self.training, p=self.dropout1)
        x_drugs_embed = self.drugs_layer_1(x_drugs_embed)

        x_sides_embed = F.relu(self.sides_bn(self.sides_layer(batch_side_features)), inplace=True)
        x_sides_embed = F.dropout(x_sides_embed, training=self.training, p=self.dropout1)
        x_sides_embed = self.sides_layer_1(x_sides_embed)
        
        # Store original embeddings for HAN fusion
        x_drugs_embed_orig = x_drugs_embed
        x_sides_embed_orig = x_sides_embed
        
        # ----------------------------------------------------------------
        # Step 2.5: Heterogeneous GNN (HAN) processing
        # ----------------------------------------------------------------
        if self.use_heterogeneous_gnn and self.subgraph_sampler is not None:
            # Sample K-hop subgraph from batch nodes
            subgraph_edge_index, node_mapping, subgraph_metadata = self.subgraph_sampler.sample(
                drug_indices, side_indices
            )#node_mapping局部节点id->全局
            subgraph_edge_index = subgraph_edge_index.to(device)
            
            # Build node features for subgraph
            reverse_mapping = {global_idx: sub_idx for sub_idx, global_idx in node_mapping.items()}
            num_subgraph_nodes = len(node_mapping)
            
            # 优化：只编码子图中实际需要的drug和adr节点，而不是全局所有节点
            # 从node_mapping中提取子图包含的drug和adr的全局索引
            subgraph_global_indices = torch.tensor([node_mapping[i] for i in range(num_subgraph_nodes)], 
                                                   dtype=torch.long, device=device)
            
            # 分离出drug和adr节点的全局索引
            drug_mask_global = subgraph_global_indices < self.graph_builder.gene_offset
            adr_mask_global = subgraph_global_indices >= self.graph_builder.adr_offset
            
            # 只对子图中的drug节点编码
            if drug_mask_global.any():
                drug_global_in_subgraph = subgraph_global_indices[drug_mask_global]
                drug_feats_subset = global_drug_features[drug_global_in_subgraph]
                x_drugs_subgraph_encoded = F.relu(self.drugs_bn(self.drugs_layer(drug_feats_subset)), inplace=True)
                x_drugs_subgraph_encoded = F.dropout(x_drugs_subgraph_encoded, training=self.training, p=self.dropout1)
                x_drugs_subgraph_encoded = self.drugs_layer_1(x_drugs_subgraph_encoded)
            
            # 只对子图中的adr节点编码
            if adr_mask_global.any():
                adr_global_in_subgraph = subgraph_global_indices[adr_mask_global] - self.graph_builder.adr_offset
                adr_feats_subset = global_side_features[adr_global_in_subgraph]
                x_sides_subgraph_encoded = F.relu(self.sides_bn(self.sides_layer(adr_feats_subset)), inplace=True)
                x_sides_subgraph_encoded = F.dropout(x_sides_subgraph_encoded, training=self.training, p=self.dropout1)
                x_sides_subgraph_encoded = self.sides_layer_1(x_sides_subgraph_encoded)
            
            # Initialize subgraph node features
            subgraph_features = torch.zeros(num_subgraph_nodes, self.embed_dim, device=device)
            gene_embedding_table = self.gene_embedding.weight if self.gene_embedding is not None else None
            
            # 直接填充特征（已经是子图局部索引顺序）mask只是标记位置的布尔变量
            if drug_mask_global.any():
                subgraph_features[drug_mask_global] = x_drugs_subgraph_encoded
            if adr_mask_global.any():
                subgraph_features[adr_mask_global] = x_sides_subgraph_encoded
            
            # 填充gene节点特征
            gene_mask_global = (~drug_mask_global) & (~adr_mask_global)
            if gene_mask_global.any() and gene_embedding_table is not None:
                gene_global_in_subgraph = subgraph_global_indices[gene_mask_global] - self.graph_builder.gene_offset
                subgraph_features[gene_mask_global] = gene_embedding_table[gene_global_in_subgraph]
            
            # 构建节点类型tensor
            node_type_tensor = torch.zeros(num_subgraph_nodes, dtype=torch.long, device=device)
            node_type_tensor[gene_mask_global] = 1  # gene
            node_type_tensor[adr_mask_global] = 2   # adr
            # drug默认为0
            
            # 构建edge_dict: 按边类型分组 {0: drug->gene, 1: gene->adr, 2: gene->drug, 3: adr->gene}
            edge_dict = {}
            if subgraph_edge_index.size(1) > 0:
                src, dst = subgraph_edge_index[0], subgraph_edge_index[1]
                src_types = node_type_tensor[src]
                dst_types = node_type_tensor[dst]
                
                # 4种边类型
                masks = {
                    0: (src_types == 0) & (dst_types == 1),  # drug->gene
                    1: (src_types == 1) & (dst_types == 2),  # gene->adr
                    2: (src_types == 1) & (dst_types == 0),  # gene->drug
                    3: (src_types == 2) & (dst_types == 1),  # adr->gene
                }
                for edge_type, mask in masks.items():
                    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if indices.numel() > 0:
                        edge_dict[edge_type] = subgraph_edge_index[:, indices]
            
            # HAN编码: 按入边类型分别聚合，只对gene节点做语义融合
            current_features = self.han_encoder(
                subgraph_features,
                edge_dict,
                self.edge_embedding,
                node_types=node_type_tensor
            )
            
            # 最终特征
            subgraph_features_han = current_features
            
            # 向量化：构建反向映射
            reverse_mapping_tensor = torch.full((self.graph_builder.get_total_nodes(),), -1, dtype=torch.long)#total_nodes足够覆盖global_indices范围
            global_indices = torch.tensor(list(reverse_mapping.keys()), dtype=torch.long)
            sub_indices = torch.tensor(list(reverse_mapping.values()), dtype=torch.long)
            reverse_mapping_tensor[global_indices] = sub_indices#将reverse_mapping字典转为向量化形式便于查询
            
            # 提取Drug和ADR节点特征（批量向量化优化）
            # 预计算全局索引和有效性掩码
            drug_global_indices = drug_indices  # 已经是全局索引[batch_size]
            adr_global_indices = self.graph_builder.adr_offset + side_indices  # [batch_size]
            
            # 批量查找子图索引（避免重复的CPU-GPU传输）
            all_global_indices = torch.cat([drug_global_indices, adr_global_indices])
            all_sub_indices = reverse_mapping_tensor[all_global_indices.cpu()].to(device)
            
            # 分离drug和adr的子图索引
            batch_size = drug_indices.size(0)
            drug_sub_indices = all_sub_indices[:batch_size]
            adr_sub_indices = all_sub_indices[batch_size:]
            
            # 计算有效性掩码
            drug_valid_mask = drug_sub_indices >= 0
            adr_valid_mask = adr_sub_indices >= 0
            
            # 批量提取HAN特征（使用掩码索引避免无效访问）
            # 对于无效索引，用0替代，后续会被原始特征覆盖
            safe_drug_indices = torch.where(drug_valid_mask, drug_sub_indices, 0)
            safe_adr_indices = torch.where(adr_valid_mask, adr_sub_indices, 0)
            batch_drug_han = torch.where(
                drug_valid_mask.unsqueeze(-1),
                subgraph_features_han[safe_drug_indices],
                x_drugs_embed_orig
            )
            
            batch_adr_han = torch.where(
                adr_valid_mask.unsqueeze(-1),
                subgraph_features_han[safe_adr_indices],
                x_sides_embed_orig
            )
            
            # Fuse HAN features with original features
            x_drugs_embed = self.han_fusion(torch.cat([x_drugs_embed_orig, batch_drug_han], dim=-1))
            x_sides_embed = self.han_fusion(torch.cat([x_sides_embed_orig, batch_adr_han], dim=-1))
        
        # ----------------------------------------------------------------
        # Step 3: Project back and chunk features
        # ----------------------------------------------------------------
        
        x_drugs = self.drug_back_proj(x_drugs_embed)
        x_sides = self.side_back_proj(x_sides_embed)
        
        # ----------------------------------------------------------------
        # Chunk features for interaction map construction
        # ChemProp 已移除
        drug_chunks = x_drugs.chunk(self.drug_chunks, 1)
        side_chunks = x_sides.chunk(self.side_chunks, 1)
        
        # ----------------------------------------------------------------
        # Step 4: Encode drug chunks
        # ----------------------------------------------------------------
        x_drug1 = F.relu(self.drug1_bn(self.drug_layer1(drug_chunks[0])), inplace=True)
        x_drug1 = F.dropout(x_drug1, training=self.training, p=self.dropout1)
        x_drug1 = self.drug_layer1_1(x_drug1)
        
        x_drug2 = F.relu(self.drug2_bn(self.drug_layer2(drug_chunks[1])), inplace=True)
        x_drug2 = F.dropout(x_drug2, training=self.training, p=self.dropout1)
        x_drug2 = self.drug_layer2_1(x_drug2)
        
        drugs = [x_drug1, x_drug2]
        
        if self.drug_chunks >= 3:
            x_drug3 = F.relu(self.drug3_4_bn(self.drug_layer3_4(drug_chunks[2])), inplace=True)
            x_drug3 = F.dropout(x_drug3, training=self.training, p=self.dropout1)
            x_drug3 = self.drug_layer3_4_1(x_drug3)
            drugs.append(x_drug3)
            
            if current_drug_chunks == 4:
                x_drug4 = F.relu(self.drug3_4_bn(self.drug_layer3_4(drug_chunks[3])), inplace=True)
                x_drug4 = F.dropout(x_drug4, training=self.training, p=self.dropout1)
                x_drug4 = self.drug_layer3_4_1(x_drug4)
                drugs.append(x_drug4)
            
        # ----------------------------------------------------------------
        # Step 5: Encode side effect chunks
        # ----------------------------------------------------------------
        x_side1 = F.relu(self.side1_bn(self.side_layer1(side_chunks[0])), inplace=True)
        x_side1 = F.dropout(x_side1, training=self.training, p=self.dropout1)
        x_side1 = self.side_layer1_1(x_side1)
        
        x_side2 = F.relu(self.side2_bn(self.side_layer2(side_chunks[1])), inplace=True)
        x_side2 = F.dropout(x_side2, training=self.training, p=self.dropout1)
        x_side2 = self.side_layer2_1(x_side2)
        
        sides = [x_side1, x_side2]
        
        if self.side_chunks == 3:
            x_side3 = F.relu(self.side3_bn(self.side_layer3(side_chunks[2])), inplace=True)
            x_side3 = F.dropout(x_side3, training=self.training, p=self.dropout1)
            x_side3 = self.side_layer3_1(x_side3)
            sides.append(x_side3)
        
        # ----------------------------------------------------------------
        # Step 6: Feature interaction attention on chunks
        # ----------------------------------------------------------------
        if self.use_feature_interaction:
            # Apply attention between drug and side chunks
            # This allows learning which drug features (DGen/GE/CS/Morgan)
            # are most relevant to which side features (MESH/GDA)
            drugs, sides = self.feature_interaction_attn(drugs, sides)
        
        # ----------------------------------------------------------------
        # Step 7: Construct interaction maps
        # ----------------------------------------------------------------
        maps = []
        for i in range(len(drugs)):
            for j in range(len(sides)):
                maps.append(torch.bmm(drugs[i].unsqueeze(2), sides[j].unsqueeze(1)))
        
        interaction_map = maps[0].view((-1, 1, self.embed_dim, self.embed_dim))
        for i in range(1, len(maps)):
            interaction = maps[i].view((-1, 1, self.embed_dim, self.embed_dim))
            interaction_map = torch.cat([interaction_map, interaction], dim=1)
        
        # ----------------------------------------------------------------
        # Step 8: Process interaction maps with ARConv (Adaptive Receptive Field)
        # ----------------------------------------------------------------
        # 预降采样：[batch, n_maps, 128, 128] -> [batch, n_maps, 64, 64]
        # 显著减少 ARConv 的显存占用（约减少 75%）
        interaction_map_downsampled = self.pre_downsample(interaction_map)
        
        # ARConv: [batch, n_maps, 64, 64] -> [batch, 32, 64, 64]
        # hw_range 设置为 [1, 9]，表示卷积核大小范围在 1x1 到 9x9
        feature_map = self.arconv_interaction(
            interaction_map_downsampled, 
            epoch=epoch, 
            hw_range=[1, 9]  # 卷积核大小自适应范围
        )
        # 自适应池化: [batch, 32, 128, 128] -> [batch, 32, 2, 2]
        feature_map = self.adaptive_pool(feature_map)
        h = feature_map.view((-1, self.channel_size * 4))
        
        # ----------------------------------------------------------------
        # Step 9: Fuse features and apply contrastive learning
        # ----------------------------------------------------------------
        total = torch.cat((x_drugs_embed, h, x_sides_embed), dim=1)
        contrastive_loss = None  # 初始化变量
        if self.use_contrastive_learning and self.training:
            # Create noisy view for contrastive learning
            noise = torch.randn_like(total)
            total_noise = total + torch.sign(total) * F.normalize(noise, dim=-1) * 0.1
            total_views = [total_noise]
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_module(
                features_list=total_views,
                fused_features=total
            )
        
        # ----------------------------------------------------------------
        # Step 10: Final prediction
        # ----------------------------------------------------------------
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)
        
        classification = self.classifier2(total)
        regression = self.con_layer(total)
        
        if self.training and contrastive_loss is not None:
            return classification.squeeze(), regression.squeeze(), contrastive_loss
        return classification.squeeze(), regression.squeeze()
    