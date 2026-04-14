# DGAPred 项目架构文档

## 1. 项目概述

DGAPred 是一个基于异质图神经网络的药物-副作用预测模型，通过整合多源特征和图结构信息来预测药物可能引发的不良反应。

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              输入层                                      │
│  Drug Features: [DGen_sim, GE_sim, CS_sim]  →  concat → [521, 1563]     │
│  ADR Features:  [MESH_sim, GDA_sim]         →  concat → [4386, 8772]    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                           特征编码层                                     │
│  drugs_layer: [1563] → [128]  +  BatchNorm + ReLU + Dropout             │
│  sides_layer: [8772] → [128]  +  BatchNorm + ReLU + Dropout             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      异质图神经网络 (HAN)                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  子图采样 (KHopSubgraphSampler)                                  │   │
│  │  - 从batch的drug/adr节点出发，BFS采样2-hop邻居                   │   │
│  │  - 分层采样: 第1层90%, 第2层50%, 第3层20%                        │   │
│  │  - 自适应采样: 高度数节点获得更多邻居                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  HANEncoder (多层异质图注意力)                                    │   │
│  │  - 4种边类型: drug→gene, gene→adr, gene→drug, adr→gene          │   │
│  │  - 节点级注意力: 多头注意力 + 动态边权重学习                       │   │
│  │  - 语义级注意力: 融合gene节点的多源信息                           │   │
│  │  - 分层聚合: 加权融合各层输出                                     │   │
│  │  - Multi-Scale: local(1-hop) + medium(2-hop)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                ↓                                        │
│  han_fusion: concat([原始特征, HAN特征]) → [256] → [128]               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                        特征分块编码                                      │
│  Drug: [128] → back_proj → [1563] → chunk(3) → 3个[521]                │
│  Side: [128] → back_proj → [8772] → chunk(2) → 2个[4386]               │
│                                                                         │
│  每个chunk独立编码: [dim] → [128] + BatchNorm + ReLU                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   特征交互注意力 (FIA-DTA 2025)                          │
│  Drug chunks ←→ Side chunks 双向交叉注意力                              │
│  - drug_to_side_attn: 每个drug chunk学习关注哪些side chunks            │
│  - side_to_drug_attn: 每个side chunk学习关注哪些drug chunks            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         交互图构建                                       │
│  对每对 (drug_chunk, side_chunk) 计算外积:                              │
│  interaction_map[i,j] = drug_chunks[i] ⊗ side_chunks[j]                │
│  → [batch, 6, 128, 128] (3 drug × 2 side = 6 maps)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      ResNet 交互图处理                                   │
│  3层残差块: [6, 128, 128] → [32, 128, 128] → flatten → [128]           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    对比学习 (CCL-ASPS 2024)                              │
│  total = concat([drug_embed, resnet_out, side_embed])                  │
│  - 添加噪声生成增强视图                                                  │
│  - InfoNCE loss 拉近原始/增强表示                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          预测层                                          │
│  total_layer: [384] → [128] + ReLU + Dropout                           │
│  classifier2: [128] → [1] (分类logits, 用BCEWithLogitsLoss)            │
│  con_layer:   [128] → [1] (回归预测, 用MSELoss)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. 核心模块详解

### 3.1 异质图构建 (HeterogeneousGraphBuilder)

```
节点类型:
- Drug: 521个, 索引 [0, 520]
- Gene: ~4000个, 索引 [521, gene_offset+num_genes-1]  
- ADR:  4386个, 索引 [adr_offset, adr_offset+4385]

边类型:
- drug_gene: Drug → Gene (药物-基因相互作用)
- gene_adr:  Gene → ADR  (基因-疾病关联)
- gene_drug: Gene → Drug (反向边)
- adr_gene:  ADR → Gene  (反向边)

数据来源:
- ctd_chem_pert_gene_ixns_list.csv → drug_gene边
- ctd_gene_adr_asso_list_4386.csv  → gene_adr边
```

### 3.2 子图采样 (KHopSubgraphSampler)

```python
# 采样流程
1. 种子节点 = batch中的drug + adr节点
2. BFS扩展K-hop邻居:
   - 第1层: 采样率90%, 最大邻居数45
   - 第2层: 采样率50%, 最大邻居数25
3. 自适应采样: 重要节点(高度数)获得更多邻居
4. 限制总节点数 ≤ max_nodes (默认350)
```

### 3.3 HAN编码器

```
HANEncoder
├── input_proj: [128] → [128]
├── HANLayer × num_layers (默认3层)
│   ├── NodeLevelAttention × 4 (每种边类型一个)
│   │   ├── W: [num_heads, in_dim, out_dim] 特征变换
│   │   ├── a: [num_heads, attn_dim, 1] 注意力参数
│   │   ├── DynamicEdgeWeightLearner (可选)
│   │   └── Multi-Scale: local + medium (可选)
│   ├── SemanticLevelAttention (仅gene节点)
│   │   └── 融合来自drug和adr的信息
│   └── out_proj: [num_heads*out_dim] → [out_dim]
├── layer_norms × num_layers
└── Hierarchical Aggregation (分层聚合)
    ├── layer_weights: 可学习的层权重
    ├── layer_attention: 动态层注意力
    └── hierarchical_fusion: 最终融合
```

### 3.4 损失函数

```python
# 分类损失 (BCEWithLogitsLoss + 标签平滑)
loss_cls = BCE(logits, smooth_labels)  # smooth: 0.05

# 回归损失 (仅正样本)
loss_reg = MSE(pred[positive], target[positive])

# 对比损失 (可选)
loss_contrast = InfoNCE(original, augmented)

# 总损失
total_loss = 0.7 * loss_cls + 0.3 * loss_reg + 0.2 * loss_contrast
```

## 4. 数据流

```
输入: drug_indices [batch], side_indices [batch]
      global_drug_features [521, 1563]
      global_side_features [4386, 8772]

Step 1: 索引取特征
  batch_drug_features = global_drug_features[drug_indices]  # [batch, 1563]
  batch_side_features = global_side_features[side_indices]  # [batch, 8772]

Step 2: 全局编码
  x_drugs_embed = MLP(batch_drug_features)  # [batch, 128]
  x_sides_embed = MLP(batch_side_features)  # [batch, 128]

Step 3: HAN处理 (如果启用)
  subgraph = sample(drug_indices, side_indices)  # 采样子图
  subgraph_features = encode_subgraph_nodes()    # 编码子图节点
  han_output = HANEncoder(subgraph_features)     # HAN传播
  x_drugs_embed = fusion(x_drugs_embed, han_drug_features)
  x_sides_embed = fusion(x_sides_embed, han_adr_features)

Step 4: 分块 + 交互
  drug_chunks = chunk(back_proj(x_drugs_embed), 3)
  side_chunks = chunk(back_proj(x_sides_embed), 2)
  drug_chunks, side_chunks = FeatureInteractionAttention(...)
  interaction_maps = outer_product(drug_chunks, side_chunks)

Step 5: ResNet + 预测
  h = ResNet(interaction_maps)
  total = concat([x_drugs_embed, h, x_sides_embed])
  logits = classifier(total)
  regression = con_layer(total)

输出: logits [batch], regression [batch]
```

## 5. 文件结构

```
pythonPredict/DGAPred(Compare)/
├── src/
│   ├── main.py                 # 训练入口, 5折交叉验证
│   ├── model/
│   │   ├── model.py            # DGAPred主模型
│   │   ├── han.py              # HAN编码器, 注意力模块
│   │   ├── resnet.py           # 残差网络块
│   │   └── chemprop_encoder.py # ChemProp分子编码器(可选)
│   └── utils/
│       ├── heterogeneous_graph.py  # 图构建, 子图采样
│       ├── data_utils.py           # 数据处理工具
│       └── drug_smiles_other_csfeatures.py  # 药物特征加载
├── 2drug-2side/DGAPred/data/   # 数据目录
│   ├── mat_drug_side_useD1_useA1.csv  # 药物-副作用标签矩阵
│   ├── ctd_chem_pert_gene_ixns_list.csv  # 药物-基因关系
│   ├── ctd_gene_adr_asso_list_4386.csv   # 基因-副作用关系
│   └── output_*/               # 训练输出
└── ARCHITECTURE.md             # 本文档
```

## 6. 关键超参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| embed_dim | 128 | 嵌入维度 |
| han_layers | 3 | HAN层数 |
| han_heads | 8 | 注意力头数 |
| subgraph_k_hop | 2 | 子图采样跳数 |
| subgraph_max_nodes | 350 | 子图最大节点数 |
| dropout1 | 0.4 | HAN dropout |
| dropout2 | 0.2 | 预测层 dropout |
| label_smooth | 0.05 | 标签平滑系数 |
| contrastive_weight | 0.2 | 对比学习权重 |
| lr | 1e-3 | 学习率 |
| batch_size | 128 | 批大小 |

## 7. 创新点

1. **分层采样策略**: 不同hop层使用不同采样率，平衡局部/全局信息
2. **动态边权重学习**: 基于节点特征动态计算边的重要性
3. **分层聚合机制**: 加权融合各HAN层输出，保留多尺度信息
4. **特征交互注意力**: Drug-Side chunks间的双向交叉注意力
5. **对比学习增强**: 通过噪声增强提升表示鲁棒性
