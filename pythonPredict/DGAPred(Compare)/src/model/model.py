"""DGAPred 主模型。"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenFeatureEncoder(nn.Module):
    """对变长基因或 MESH token 序列进行池化编码。"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, token_ids, offsets, entity_indices):
        """按 batch 实体索引取变长 token，并计算平均池化表示。"""
        batch_size = len(entity_indices)
        if token_ids.numel() == 0:
            return self.embedding.weight.new_zeros((batch_size, self.embedding.embedding_dim))

        starts = offsets[entity_indices]
        ends = offsets[entity_indices + 1]
        batch_token_ids = []
        batch_offsets = []
        current_offset = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            batch_offsets.append(current_offset)
            current_tokens = token_ids[start:end]
            if current_tokens.numel() == 0:
                # padding_idx 不参与均值，空实体因此得到全零表示。
                current_tokens = token_ids.new_zeros(1)
            batch_token_ids.append(current_tokens)
            current_offset += current_tokens.numel()

        pooled = self.embedding(
            torch.cat(batch_token_ids),
            torch.as_tensor(batch_offsets, dtype=torch.long, device=token_ids.device),
        )
        return self.projection(pooled)


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
        drug_token_vocab_sizes=None,
        side_token_vocab_sizes=None,
        token_feature_mode: str = "none",
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

        self.drug_feature_dims = list(drug_feature_dims or [])
        self.side_feature_dims = list(side_feature_dims or [])
        self.token_feature_mode = token_feature_mode
        self.drug_token_vocab_sizes = list(drug_token_vocab_sizes or [])
        self.side_token_vocab_sizes = list(side_token_vocab_sizes or [])
        self.drug_chunks = len(self.drug_feature_dims)
        self.side_chunks = len(self.side_feature_dims)
        if self.token_feature_mode == "replace":
            self.drug_chunks += len(self.drug_token_vocab_sizes)
            self.side_chunks += len(self.side_token_vocab_sizes)
        # 全局拼接特征编码，用来保留原模型的整体药物/副作用表示。
        self.drugs_layer = None
        self.drugs_layer_1 = None
        self.drugs_bn = None
        if self.drugs_dim > 0:
            self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
            self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.sides_layer = None
        self.sides_layer_1 = None
        self.sides_bn = None
        if self.sides_dim > 0:
            self.sides_layer = nn.Linear(self.sides_dim, self.embed_dim)
            self.sides_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.sides_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.drug_token_encoders = nn.ModuleList([
            TokenFeatureEncoder(vocab_size, self.embed_dim)
            for vocab_size in self.drug_token_vocab_sizes
        ])
        self.side_token_encoders = nn.ModuleList([
            TokenFeatureEncoder(vocab_size, self.embed_dim)
            for vocab_size in self.side_token_vocab_sizes
        ])

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

    @staticmethod
    def _encode_token_features(token_features, encoders, entity_indices, entity_type):
        """编码当前 batch 对应的所有原始 token 特征源。"""
        if not encoders:
            return []
        if token_features is None or len(token_features) != len(encoders):
            raise ValueError(f"模型已启用 {entity_type} token 特征，但输入特征数量不匹配。")
        return [
            encoder(token_ids, offsets, entity_indices)
            for encoder, (token_ids, offsets) in zip(encoders, token_features)
        ]

    def forward(
        self,
        drug_indices: torch.Tensor,
        side_indices: torch.Tensor,
        device: torch.device,
        global_drug_features: torch.Tensor,
        global_side_features: torch.Tensor,
        global_drug_token_features=None,
        global_side_token_features=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 batch 索引取全局特征，并输出分类 logits 与回归分数。"""
        drug_indices = drug_indices.to(device, non_blocking=True)
        side_indices = side_indices.to(device, non_blocking=True)

        batch_drug_features = global_drug_features[drug_indices]
        batch_side_features = global_side_features[side_indices]

        if self.drugs_layer is None:
            x_drugs = batch_drug_features.new_zeros((len(drug_indices), self.embed_dim))
        else:
            x_drugs = F.relu(self.drugs_bn(self.drugs_layer(batch_drug_features)), inplace=True)
            x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
            x_drugs = self.drugs_layer_1(x_drugs)

        if self.sides_layer is None:
            x_sides = batch_side_features.new_zeros((len(side_indices), self.embed_dim))
        else:
            x_sides = F.relu(self.sides_bn(self.sides_layer(batch_side_features)), inplace=True)
            x_sides = F.dropout(x_sides, training=self.training, p=self.dropout1)
            x_sides = self.sides_layer_1(x_sides)

        drug_token_embeddings = self._encode_token_features(
            global_drug_token_features, self.drug_token_encoders, drug_indices, "药物"
        )
        side_token_embeddings = self._encode_token_features(
            global_side_token_features, self.side_token_encoders, side_indices, "ADR"
        )
        if drug_token_embeddings:
            x_drugs = x_drugs + torch.stack(drug_token_embeddings, dim=0).sum(dim=0)
        if side_token_embeddings:
            x_sides = x_sides + torch.stack(side_token_embeddings, dim=0).sum(dim=0)

        drug_chunks = torch.split(batch_drug_features, self.drug_feature_dims, dim=1) if self.drug_feature_dims else []
        side_chunks = torch.split(batch_side_features, self.side_feature_dims, dim=1) if self.side_feature_dims else []
        drugs = self._encode_chunks(drug_chunks, self.drug_layers, self.drug_layers_1, self.drug_bns)
        sides = self._encode_chunks(side_chunks, self.side_layers, self.side_layers_1, self.side_bns)
        if self.token_feature_mode == "replace":
            drugs.extend(drug_token_embeddings)
            sides.extend(side_token_embeddings)

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
