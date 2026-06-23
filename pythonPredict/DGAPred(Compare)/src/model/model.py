"""DGAPred 主模型。"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ARConv import ARConv


# ============================================================================
# Contrastive Learning Module
# ============================================================================

class ContrastiveLearningModule(nn.Module):
    """Collaborative Contrastive Learning Module (CCL-ASPS 2024)
    
    Implements contrastive learning between individual views and fused features
    to enhance representation quality.
    """
    def __init__(self, feature_dim: int, temperature: float = 0.07,
                 loss_type: str = "standard", tau_plus: float = 0.1):
        super(ContrastiveLearningModule, self).__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.tau_plus = tau_plus
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )

    def _standard_infonce(self, similarity: torch.Tensor) -> torch.Tensor:
        batch_size = similarity.size(0)
        labels = torch.arange(batch_size, device=similarity.device)
        return F.cross_entropy(similarity, labels)

    def _debiased_infonce(self, similarity: torch.Tensor) -> torch.Tensor:
        batch_size = similarity.size(0)
        if batch_size <= 1:
            return similarity.new_tensor(0.0)

        exp_sim = torch.exp(similarity)
        pos_sim = exp_sim.diag()
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
        neg_sum = exp_sim.masked_select(neg_mask).view(batch_size, batch_size - 1).sum(dim=1)
        neg_count = batch_size - 1

        # DCL估计并扣除负样本池中的假阴性污染。
        debiased_neg = (neg_sum - self.tau_plus * neg_count * pos_sim) / (1.0 - self.tau_plus)
        neg_floor = neg_count * torch.exp(torch.tensor(-1.0 / self.temperature, device=similarity.device))
        debiased_neg = torch.clamp(debiased_neg, min=neg_floor)
        loss = -torch.log(pos_sim / (pos_sim + debiased_neg + 1e-12))
        return loss.mean()
    
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
            
            # 计算当前视图与融合表示之间的相似度矩阵，对角线是正样本对。
            similarity = torch.matmul(view_proj, fused_proj.T) / self.temperature
            
            if self.loss_type == "debiased":
                loss = self._debiased_infonce(similarity)
            else:
                loss = self._standard_infonce(similarity)
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
                 contrastive_loss_type: str = "standard",
                 d4_tau_plus: float = 0.1):
        """Initialize DGAPred model.
        
        Args:
            drugs_dim: Input dimension for drug features
            sides_dim: Input dimension for side effect features
            embed_dim: Embedding dimension
            batchsize: Batch size
            dropout1: Encoder dropout rate
            dropout2: Final prediction dropout rate
            n_drug_chunks: Number of chunks for drug features
            n_side_chunks: Number of chunks for side effect features
            use_feature_interaction: Whether to use feature interaction attention
            use_contrastive_learning: Whether to use contrastive learning
            contrastive_loss_type: 对比损失类型，standard或debiased
            d4_tau_plus: DCL假阴性比例先验
        """
        super(DGAPred, self).__init__()
        
        # 基础维度配置
        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.embed_dim = embed_dim
        self.batchsize = batchsize
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        
        # 多源特征分块配置
        self.drug_chunks = n_drug_chunks
        self.side_chunks = n_side_chunks
        self.drug_dim = drugs_dim // self.drug_chunks
        self.side_dim = sides_dim // self.side_chunks
        
        # 高级模块开关
        self.use_feature_interaction = use_feature_interaction
        self.use_contrastive_learning = use_contrastive_learning
        
        # ----------------------------------------------------------------
        # 全局特征编码层
        # ----------------------------------------------------------------
        self.drugs_layer = nn.Linear(drugs_dim, embed_dim)
        self.drugs_layer_1 = nn.Linear(embed_dim, embed_dim)
        self.drugs_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        self.sides_layer = nn.Linear(sides_dim, embed_dim)
        self.sides_layer_1 = nn.Linear(embed_dim, embed_dim)
        self.sides_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        # 投影回原始维度后再切块，保持当前 D1 主逻辑不变。
        self.drug_back_proj = nn.Linear(embed_dim, drugs_dim)
        self.side_embed_dim = embed_dim  # side embedding 要走图的drug->side传播涉及concat特殊处理
        
        # ----------------------------------------------------------------
        # 分块特征编码层
        # ----------------------------------------------------------------

        # 药物分块编码器
        self.drug_layer1 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer1_1 = nn.Linear(embed_dim, embed_dim)
        self.drug_layer2 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer2_1 = nn.Linear(embed_dim, embed_dim)
        self.drug_layer3_4 = nn.Linear(self.drug_dim, embed_dim)
        self.drug_layer3_4_1 = nn.Linear(embed_dim, embed_dim)
        
        self.drug1_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.drug2_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        self.drug3_4_bn = nn.BatchNorm1d(embed_dim, momentum=0.5)
        
        # 副作用分块编码器
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
        # 交互图处理模块
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
        
        # 副作用回投影层
        self.side_back_proj = nn.Linear(self.side_embed_dim, self.sides_dim)
        
        # ----------------------------------------------------------------
        # 高级模块
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
                temperature=0.07,
                loss_type=contrastive_loss_type,
                tau_plus=d4_tau_plus
            )
        
        # ChemProp 编码器已移除
            
        # ----------------------------------------------------------------
        # 最终预测层
        # ----------------------------------------------------------------
        
        total_input_dim = self.channel_size * 4 + embed_dim + self.side_embed_dim
        self.total_layer = nn.Linear(total_input_dim, self.channel_size * 4)
        self.classifier2 = nn.Linear(self.channel_size * 4, 1)  # Outputs logits
        self.con_layer = nn.Linear(self.channel_size * 4, 1)


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
            
            # 这里判断的是模型当前配置的药物分块数，回退时残留了错误变量名
            if self.drug_chunks == 4:
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
            # 构造带轻微噪声的增强视图，用于和融合表示做对比学习。
            noise = torch.randn_like(total)
            total_noise = total + torch.sign(total) * F.normalize(noise, dim=-1) * 0.1
            total_views = [total_noise]
            
            # 根据开关选择标准 InfoNCE 或 D4 去偏 InfoNCE。
            contrastive_loss = self.contrastive_module(
                features_list=total_views,
                fused_features=total
            )
        
        # ----------------------------------------------------------------
        # Step 10: 最终预测
        # ----------------------------------------------------------------
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)
        
        classification = self.classifier2(total)
        regression = self.con_layer(total)
        
        if self.training and contrastive_loss is not None:
            return classification.squeeze(), regression.squeeze(), contrastive_loss
        return classification.squeeze(), regression.squeeze()
    
