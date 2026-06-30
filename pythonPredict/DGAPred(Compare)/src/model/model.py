"""DGAPred 主模型。"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Main DGAPred Model
# ============================================================================

class DGAPred(nn.Module):
    """干净版 DGAPred：全局特征索引、CNN 交互图和 dropout 正则。"""
    
    def __init__(self, drugs_dim: int, sides_dim: int, embed_dim: int=128, 
                 batchsize: int=128, dropout1: float = 0.5, dropout2: float = 0.5,
                 n_drug_chunks: int = 2, n_side_chunks: int = 2):
        """初始化模型，只保留已验证有效的基础结构和正则化参数。"""
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
        # 多源药物特征和副作用特征两两构造交互图。
        self.number_map = self.drug_chunks * self.side_chunks
        
        # 固定使用 DGANet-main 原始 6 层 CNN 交互图，不再保留其它算法分支。
        self.cnn_interaction = nn.Sequential(
            nn.Conv2d(self.number_map, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.BatchNorm2d(self.channel_size),
            nn.ReLU(),
        )

        # 副作用回投影层
        self.side_back_proj = nn.Linear(self.side_embed_dim, self.sides_dim)
            
        # ----------------------------------------------------------------
        # 最终预测层
        # ----------------------------------------------------------------
        
        total_input_dim = self.channel_size * 4 + embed_dim + self.side_embed_dim
        self.total_layer = nn.Linear(total_input_dim, self.channel_size * 4)
        self.classifier2 = nn.Linear(self.channel_size * 4, 1)  # Outputs logits
        self.con_layer = nn.Linear(self.channel_size * 4, 1)


    def forward(self, drug_indices: torch.Tensor, side_indices: torch.Tensor, 
                device: torch.device, global_drug_features: torch.Tensor, 
                global_side_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 batch 索引取全局特征，并输出分类 logits 与回归分数。"""
        # ----------------------------------------------------------------
        # Step 1: 取出 batch 对应的全局特征
        # ----------------------------------------------------------------
        # 全局特征矩阵已经在外层按 fold 搬到目标设备，这里只搬很小的索引张量。
        drug_indices = drug_indices.to(device, non_blocking=True)
        side_indices = side_indices.to(device, non_blocking=True)
        
        batch_drug_features = global_drug_features[drug_indices]
        batch_side_features = global_side_features[side_indices]
        
        # ----------------------------------------------------------------
        # Step 2: 编码 batch 特征
        # ----------------------------------------------------------------
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
        # Step 6: 构造交互图
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
        # Step 7: 使用原始 CNN 处理交互图
        # ----------------------------------------------------------------
        feature_map = self.cnn_interaction(interaction_map)
        h = feature_map.view((-1, self.channel_size * 4))
        
        # ----------------------------------------------------------------
        # Step 8: 融合特征并预测
        # ----------------------------------------------------------------
        total = torch.cat((x_drugs_embed, h, x_sides_embed), dim=1)
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)
        
        classification = self.classifier2(total)
        regression = self.con_layer(total)
        
        return classification.squeeze(), regression.squeeze()
    
