"""DGAPred 主模型。"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGAPred(nn.Module):
    """干净版 DGAPred：支持任意数量特征的 CNN 交互图模型。"""

    def __init__(
        self,
        drugs_dim: int,
        sides_dim: int,
        embed_dim: int = 128,
        batchsize: int = 128,
        dropout1: float = 0.5,
        dropout2: float = 0.5,
        drug_feature_dims=None,
        side_feature_dims=None,
    ):
        """初始化模型。

        drug_feature_dims 和 side_feature_dims 记录每个特征矩阵的宽度，
        这样开关任意特征后，模型仍能按真实维度切分和编码。
        """
        super(DGAPred, self).__init__()
        self.drugs_dim = drugs_dim
        self.sides_dim = sides_dim
        self.batchsize = batchsize
        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2

        self.drug_feature_dims = list(drug_feature_dims or [drugs_dim])
        self.side_feature_dims = list(side_feature_dims or [sides_dim])
        self.drug_chunks = len(self.drug_feature_dims)
        self.side_chunks = len(self.side_feature_dims)

        # 全局拼接特征编码，用来保留原模型的整体药物/副作用表示。
        self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.sides_layer = nn.Linear(self.sides_dim, self.embed_dim)
        self.sides_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.sides_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        # 每个特征源单独编码，避免因为特征数量变化导致维度写死。
        self.drug_layers = nn.ModuleList()
        self.drug_layers_1 = nn.ModuleList()
        self.drug_bns = nn.ModuleList()
        for feature_dim in self.drug_feature_dims:
            self.drug_layers.append(nn.Linear(feature_dim, self.embed_dim))
            self.drug_layers_1.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.drug_bns.append(nn.BatchNorm1d(self.embed_dim, momentum=0.5))

        self.side_layers = nn.ModuleList()
        self.side_layers_1 = nn.ModuleList()
        self.side_bns = nn.ModuleList()
        for feature_dim in self.side_feature_dims:
            self.side_layers.append(nn.Linear(feature_dim, self.embed_dim))
            self.side_layers_1.append(nn.Linear(self.embed_dim, self.embed_dim))
            self.side_bns.append(nn.BatchNorm1d(self.embed_dim, momentum=0.5))

        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.number_map = self.drug_chunks * self.side_chunks

        # 固定使用 DGANet-main 的 6 层 CNN 交互图。
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

        total_input_dim = self.channel_size * 4 + 2 * self.embed_dim
        self.total_layer = nn.Linear(total_input_dim, self.channel_size * 4)
        self.classifier2 = nn.Linear(self.channel_size * 4, 1)
        self.con_layer = nn.Linear(self.channel_size * 4, 1)

    def _encode_chunks(self, chunks, layers, layers_1, bns):
        """逐个编码特征源，返回统一 embed_dim 的特征列表。"""
        outputs = []
        for chunk, layer, layer_1, bn in zip(chunks, layers, layers_1, bns):
            x = F.relu(bn(layer(chunk)), inplace=True)
            x = F.dropout(x, training=self.training, p=self.dropout1)
            outputs.append(layer_1(x))
        return outputs

    def forward(
        self,
        drug_indices: torch.Tensor,
        side_indices: torch.Tensor,
        device: torch.device,
        global_drug_features: torch.Tensor,
        global_side_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 batch 索引取全局特征，并输出分类 logits 与回归分数。"""
        drug_indices = drug_indices.to(device, non_blocking=True)
        side_indices = side_indices.to(device, non_blocking=True)

        batch_drug_features = global_drug_features[drug_indices]
        batch_side_features = global_side_features[side_indices]

        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(batch_drug_features)), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        x_sides = F.relu(self.sides_bn(self.sides_layer(batch_side_features)), inplace=True)
        x_sides = F.dropout(x_sides, training=self.training, p=self.dropout1)
        x_sides = self.sides_layer_1(x_sides)

        drug_chunks = torch.split(batch_drug_features, self.drug_feature_dims, dim=1)
        side_chunks = torch.split(batch_side_features, self.side_feature_dims, dim=1)
        drugs = self._encode_chunks(drug_chunks, self.drug_layers, self.drug_layers_1, self.drug_bns)
        sides = self._encode_chunks(side_chunks, self.side_layers, self.side_layers_1, self.side_bns)

        maps = []
        for drug_feature in drugs:
            for side_feature in sides:
                maps.append(torch.bmm(drug_feature.unsqueeze(2), side_feature.unsqueeze(1)))

        interaction_map = torch.cat(
            [item.view((-1, 1, self.embed_dim, self.embed_dim)) for item in maps],
            dim=1,
        )
        feature_map = self.cnn_interaction(interaction_map)
        h = feature_map.view((-1, self.channel_size * 4))

        total = torch.cat((x_drugs, h, x_sides), dim=1)
        total = F.relu(self.total_layer(total), inplace=True)
        total = F.dropout(total, training=self.training, p=self.dropout2)

        classification = self.classifier2(total)
        regression = self.con_layer(total)
        return classification.squeeze(), regression.squeeze()
