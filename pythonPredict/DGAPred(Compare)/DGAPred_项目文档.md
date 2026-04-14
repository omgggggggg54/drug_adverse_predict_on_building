# DGAPred 项目文档

## 项目概述

DGAPred（Drug-Gene-ADR Prediction）是一个基于异质图神经网络的药物不良反应预测模型。该项目采用多模态特征融合、异质图学习和对比学习等先进技术，用于预测药物与其不良反应（ADR）之间的关联。

---

## 1. 数据来源

### 1.1 主要数据集

| 数据源 | 文件名 | 描述 |
|------|--------|------|
| **药物-不良反应** | `sider_pert_mesh_list.csv` | SIDER数据库中的药物-MESH不良反应对应关系 |
| **药物-基因相互作用** | `ctd_chem_pert_gene_ixns_list.csv` | CTD数据库中的化学物质-基因相互作用 |
| **基因-不良反应关联** | `ctd_gene_adr_asso_list_4386.csv` | CTD数据库中的基因-疾病（不良反应）关联 |
| **药物列表** | `lincs_druglist_ge_go_521.csv` | LINCS数据库中的521个药物列表 |

### 1.2 相似性矩阵

预计算的相似性矩阵存储在 `similarity_path` 目录中：

| 矩阵名称 | 维度 | 描述 |
|---------|------|------|
| `mat_drug_cs_similarity.csv` | (n_drugs, n_drugs) | 药物化学结构相似性（Tanimoto相似度） |
| `mat_drug_dgen_similarity.csv` | (n_drugs, n_drugs) | 药物-基因相互作用相似性 |
| `mat_drug_ge_similarity.csv` | (n_drugs, n_drugs) | 药物基因表达相似性 |
| `mat_adr_mesh_similarity.csv` | (n_adrs, n_adrs) | 不良反应MESH本体相似性 |
| `mat_adr_GDisease_similarity.csv` | (n_adrs, n_adrs) | 不良反应基因-疾病关联相似性 |

### 1.3 分子特征

- **ChemProp特征**：基于SMILES字符串生成的分子表征
  - 邻接矩阵（adj）
  - 距离矩阵（dist）
  - 化学键类型（clb）
  - Morgan指纹特征

---

## 2. 数据处理

### 2.1 标签矩阵构建

```
load_label(screen_drug_list, use_DGen, use_AGen, args)
```

**处理流程：**

1. **缓存检查**：检查是否存在预计算的标签矩阵缓存
   - 缓存文件格式：`mat_drug_side_useD{0/1}_useA{0/1}.csv`

2. **数据加载**：从SIDER数据库加载原始药物-不良反应对应关系

3. **药物过滤**（可选）：
   - 若 `use_DGen=True`：仅保留在CTD药物-基因相互作用数据库中出现的药物
   - 若 `use_AGen=True`：仅保留在CTD基因-疾病关联数据库中出现的不良反应

4. **矩阵转换**：将三元组列表 `[drug_id, adr_id, label]` 转换为矩阵格式
   - 行：药物ID
   - 列：不良反应ID
   - 值：标签（0/1）

5. **缓存保存**：将处理后的矩阵保存到本地

**输出：**
- `drug_list`：药物ID列表
- `adr_list`：不良反应ID列表
- `drug_side`：药物-不良反应标签矩阵 (n_drugs × n_adrs)

### 2.2 样本提取与平衡

```
Extract_positive_negative_samples(DAL, addition_negative_number='all')
```

**处理流程：**

1. **矩阵展平**：将 (n_drugs × n_adrs) 的矩阵转换为 (n_samples, 3) 的三元组列表
   - 格式：`[drug_idx, adr_idx, label]`

2. **样本分离**：按标签排序，分离正样本和负样本
   - 正样本：标签为1的样本
   - 负样本：标签为0的样本

3. **负样本采样**：
   - 若 `addition_negative_number='all'`：采样所有负样本
   - 若为整数：采样 `(1 + addition_negative_number) * n_positive` 个负样本

4. **样本平衡**：
   - 最终正样本数 = 最终负样本数
   - 额外负样本用于验证集

**输出：**
- `addition_negative_sample`：额外负样本（用于验证）
- `final_positive_sample`：平衡后的正样本
- `final_negative_sample`：平衡后的负样本

### 2.3 药物特征加载

```
load_drug_feature(screen_drug_list, args)
```

**加载的特征类型：**

| 特征类型 | 来源 | 维度 | 描述 |
|---------|------|------|------|
| DGen | CTD药物-基因相互作用 | (n_drugs, n_drugs) | 药物基因相互作用相似性 |
| GE | 基因表达 | (n_drugs, n_drugs) | 药物诱导的基因表达相似性 |
| CS | 化学结构 | (n_drugs, n_drugs) | 分子结构相似性 |
| Morgan | SMILES指纹 | (n_drugs, n_drugs) | Morgan指纹Tanimoto相似性 |

**输出：** 特征矩阵列表 `[drug_DGen, drug_ge_sim, drug_cs_sim]`

### 2.4 不良反应特征加载

```
load_adr_feature(screen_adr_list, args)
```

**加载的特征类型：**

| 特征类型 | 来源 | 维度 | 描述 |
|---------|------|------|------|
| MESH | MESH本体 | (n_adrs, n_adrs) | 不良反应MESH层级相似性 |
| GDA | 基因-疾病关联 | (n_adrs, n_adrs) | 不良反应基因关联相似性 |

**输出：** 特征矩阵列表 `[side_mesh_sim, adr_GDisease_sim]`

---

## 3. 数据输入

### 3.1 输入数据格式

**训练/测试数据：**
```
data = [(drug_idx, adr_idx, label), ...]
```
- `drug_idx`：药物在药物列表中的索引 (0 ~ n_drugs-1)
- `adr_idx`：不良反应在不良反应列表中的索引 (0 ~ n_adrs-1)
- `label`：二分类标签 (0/1) 或回归标签 (0.0 ~ 1.0)

### 3.2 批处理

**DataLoader配置：**
- 批大小：128（训练）/ 128（测试）
- 采样方式：随机采样
- 数据格式：`(drug_indices, adr_indices, ratings)`

**批处理流程：**

1. 从批数据中提取药物索引和不良反应索引
2. 根据全局特征矩阵查询对应的特征向量
3. 构建异质图子图（K-hop采样）
4. 输入模型进行前向传播

### 3.3 全局特征矩阵

在训练/测试前，将所有特征矩阵转换为全局特征张量：

```python
# 药物特征：[n_drugs, sum(drug_feature_dims)]
global_drug_features = concatenate([drug_DGen, drug_ge, drug_cs])

# 不良反应特征：[n_adrs, sum(adr_feature_dims)]
global_side_features = concatenate([adr_mesh, adr_gdisease])
```

---

## 4. 模型架构

### 4.1 整体架构

```
DGAPred Model
├── 输入层
│   ├── 药物特征编码 (drugs_dim -> embed_dim)
│   └── 不良反应特征编码 (sides_dim -> embed_dim)
├── 异质图神经网络 (HAN)
│   ├── K-hop子图采样
│   ├── 多层HAN编码
│   └── 节点特征融合
├── 特征交互注意力 (FIA-DTA 2025)
│   ├── 药物特征分块
│   ├── 不良反应特征分块
│   └── 交叉模态注意力
├── 交互图构建
│   ├── 特征块编码
│   ├── 交互矩阵生成
│   └── ResNet处理
├── 对比学习 (CCL-ASPS 2024)
│   ├── 多视图特征提取
│   └── 对比损失计算
└── 预测头
    ├── 二分类输出 (classification)
    └── 回归输出 (regression)
```

### 4.2 核心模块详解

#### 4.2.1 特征编码层

**药物特征编码：**
```
输入: global_drug_features [batch_size, drugs_dim]
  ↓
drugs_layer: Linear(drugs_dim, embed_dim)
  ↓
drugs_bn: BatchNorm1d(embed_dim)
  ↓
ReLU激活
  ↓
Dropout(dropout1)
  ↓
drugs_layer_1: Linear(embed_dim, embed_dim)
  ↓
输出: x_drugs_embed [batch_size, embed_dim]
```

**不良反应特征编码：** 类似药物特征编码

#### 4.2.2 异质图神经网络 (HAN)

本节详细说明异质图神经网络的多尺度注意力机制中向量的具体变化。

---

**一、图结构定义**

- **节点类型**：3种
  - Drug (0)：药物节点，共 n_drugs 个
  - Gene (1)：基因节点，共 n_genes 个
  - ADR (2)：不良反应节点，共 n_adrs 个

- **边类型**：4种
  - 0: Drug → Gene（正向）
  - 1: Gene → ADR（正向）
  - 2: Gene → Drug（反向）
  - 3: ADR → Gene（反向）

---

**二、子图采样与节点特征初始化**

给定一个batch的药物索引和不良反应索引，首先进行K-hop子图采样：
- 从batch中的药物和ADR节点出发，BFS采样K跳邻居（默认K=3）
- 得到子图节点集合，包含Drug、Gene、ADR三类节点
- 子图最大节点数限制为300个
- 自适应采样：根据节点度中心性，重要节点采样更多邻居（最多50个）

**节点特征初始化：**
- **Drug节点**：使用编码后的药物特征向量，维度为 [embed_dim]（如128维）
- **ADR节点**：使用编码后的不良反应特征向量，维度为 [embed_dim]
- **Gene节点**：使用可学习的嵌入表（Embedding Table），每个基因有独立的128维向量

假设子图有N个节点，初始特征矩阵为 **X**，形状 [N, 128]。

---

**三、节点级注意力（Node-Level Attention）**

对于每种边类型（共4种），分别计算注意力并聚合邻居信息：
- 边类型0：Drug → Gene
- 边类型1：Gene → ADR  
- 边类型2：Gene → Drug
- 边类型3：ADR → Gene

**步骤3.1：特征线性变换**

每个节点的特征向量经过线性变换，这一步的目的是将节点特征投影到注意力空间，学习节点的隐藏表示。

设节点 i 的原始特征为 **x_i**（128维向量），经过权重矩阵 **W** 变换：

- **W的形状**：[num_heads, in_dim, out_dim] = [4, 128, 128]
- **数学意义**：W是可学习的投影矩阵，每个注意力头有独立的W，使不同头可以关注特征空间的不同方面

> **h_i^(k)** = **x_i** × **W^(k)**  （对于第k个头）

结果：每个节点在每个头下得到128维的变换后特征。4个头独立计算，每个节点最终有 [4, 128] 的特征张量。

**步骤3.2：注意力分数计算**

对于每条边 (src → dst)，计算该边的注意力分数，决定源节点对目标节点的重要程度：

1. 取出源节点变换后特征 **h_src** 和目标节点变换后特征 **h_dst**，各128维
2. 获取边类型嵌入 **e_edge**（32维，从EdgeEmbedding查表得到，表示边的类型信息）
3. 拼接三者形成288维向量：
   > **concat** = [**h_src**, **h_dst**, **e_edge**]
   
   维度计算：128 + 128 + 32 = 288维

4. 通过注意力向量 **a** 计算原始分数：
   - **a的形状**：[num_heads, 288, 1]
   - **数学意义**：a是可学习的注意力权重向量，用于将拼接后的特征映射为标量分数。它学习「什么样的源-目标-边组合应该获得更高的注意力」

   > raw_score = **concat** × **a**（矩阵乘法，288维向量 × [288,1] = 标量）

5. 应用LeakyReLU激活（允许负值有小梯度，防止神经元死亡）：
   > score = LeakyReLU(raw_score, slope=0.2)

**步骤3.3：Softmax归一化**

对于每个目标节点dst，将所有指向它的边的分数做Softmax归一化，使注意力权重和为1：

> α_ij = exp(score_ij) / Σ_k exp(score_kj)

其中 j 是目标节点，i 和 k 是所有指向 j 的源节点。

**数学意义**：Softmax确保所有邻居的注意力权重之和为1，形成概率分布，表示每个邻居对目标节点的相对重要性。

**步骤3.4：邻居特征加权聚合**

目标节点 j 的新特征等于所有邻居特征的加权和：

> **h_j'** = Σ_i (α_ij × **h_i**)

即：每个邻居的128维特征向量乘以对应的注意力权重（标量），然后所有邻居逐元素相加。

**数学意义**：这是消息传递的核心——每个节点聚合其邻居的信息，重要的邻居贡献更大（权重α更高），不重要的邻居贡献更小。

**多头合并**：4个头各产生128维输出，拼接得到512维，再通过输出投影层压缩回128维：
> **z_j** = OutputProj([**h_j'^(1)**, **h_j'^(2)**, **h_j'^(3)**, **h_j'^(4)**])

**多头的数学意义**：不同的头学习不同的注意力模式。例如，一个头可能关注结构相似性，另一个头可能关注功能相关性，最终融合多种视角的信息。

---

**四、多尺度注意力（Local + Medium）**

当启用多尺度注意力时，同时使用两个尺度：

**Local尺度（1-hop）**：使用原始邻接关系
- 直接使用上述节点级注意力
- 输出 **z_local**，形状 [N, 128]

**Medium尺度（2-hop）**：使用2跳邻接关系
- 计算邻接矩阵的平方 **A²**，找出所有2跳可达的节点对
- 在2跳邻接图上执行同样的节点级注意力
- 输出 **z_medium**，形状 [N, 128]

**2跳邻接矩阵计算**：
> **A²** = **A** × **A**

其中 **A** 是原始稀疏邻接矩阵。**A²**[i][j] > 0 表示节点 i 和 j 之间存在长度为2的路径。

**尺度融合**：

将两个尺度的输出拼接后通过线性层融合：

1. 拼接：**z_concat** = [**z_local**, **z_medium**]，形状 [N, 256]
2. 线性变换：**z_fused** = **z_concat** × **W_fusion** + **b**，输出 [N, 128]
3. LayerNorm归一化：**z_out** = LayerNorm(**z_fused**)

---

**五、语义级注意力（Semantic-Level Attention）**

**仅对Gene节点**进行语义融合，因为Gene节点同时接收来自Drug和ADR的信息：

假设某个Gene节点从Drug→Gene边得到特征 **z_from_drug**（128维），从ADR→Gene边得到特征 **z_from_adr**（128维）。

**步骤5.1：计算每种入边的重要性分数**

对于入边类型 i（i=0表示来自Drug，i=1表示来自ADR）：

1. 线性变换：**t_i** = **z_i** × **W_i**，其中 **W_i** 是该边类型专属的权重矩阵
2. 与查询向量点积：score_i = **t_i** · **q_i**（得到标量）
3. LeakyReLU激活

**步骤5.2：Softmax归一化**

> β_0 = exp(score_0) / (exp(score_0) + exp(score_1))
> β_1 = exp(score_1) / (exp(score_0) + exp(score_1))

**步骤5.3：加权融合**

> **z_gene** = β_0 × **z_from_drug** + β_1 × **z_from_adr**

即两个128维向量分别乘以各自的权重（标量），然后逐元素相加，得到融合后的128维向量。

---

**六、多层HAN与残差连接**

堆叠多个HAN层（默认3层），每层之间使用残差连接：

设第 l 层输入为 **h^(l)**，输出为 **h'^(l)**：

> **h^(l+1)** = LayerNorm(**h^(l)** + **h'^(l)**)

即：将层输入与层输出逐元素相加，然后归一化。这样梯度可以直接回传，防止梯度消失。

---

**七、HAN输出与原始特征融合**

从HAN编码后的子图中，提取batch中Drug和ADR节点的特征：

- **h_drug_han**：batch中药物节点经HAN编码后的128维特征
- **h_adr_han**：batch中ADR节点经HAN编码后的128维特征

与原始编码特征融合：

1. 拼接：**concat_drug** = [**x_drug_orig**, **h_drug_han**]，形状 [batch, 256]
2. 线性变换：**x_drug_final** = **concat_drug** × **W_fusion** + **b**，输出 [batch, 128]

ADR特征同理处理。

#### 4.2.3 特征交互注意力 (FIA-DTA 2025)

本模块让药物的不同特征类型（如DGen/GE/CS）与不良反应的不同特征类型（如MESH/GDA）进行交叉注意力交互，学习哪些特征组合对预测最重要。

---

**一、特征分块与编码**

**步骤1：投影回原始维度**

HAN编码后的Drug和ADR特征（各128维）先投影回原始维度：

> **x_drugs_proj** = **x_drugs_embed** × **W_back_proj**

形状变化：[batch, 128] → [batch, drugs_dim]（如drugs_dim=256或512，取决于输入特征数量）

**步骤2：均匀切分**

使用`chunk()`将投影后的特征均匀切分成多个块：

- Drug特征切分为 n_drug_chunks 块（默认2-3块，若启用ChemProp则+1块）
- ADR特征切分为 n_side_chunks 块（默认2块）

假设 drugs_dim=512，n_drug_chunks=2，则每块维度 = 512/2 = 256

**步骤3：独立编码器**

每个特征块通过独立的编码网络变换到embed_dim（128维）：

> **x_drug_i** = Linear_2(Dropout(ReLU(BatchNorm(Linear_1(**chunk_i**)))))

形状变化：[batch, drug_dim] → [batch, 128]

**最终得到**：
- drugs列表：[**x_drug1**, **x_drug2**, ...]，每个128维
- sides列表：[**x_side1**, **x_side2**]，每个128维

---

**二、交叉模态注意力**

将特征块列表转换为序列，进行双向交叉注意力：

**步骤1：堆叠为序列**

> **drug_seq** = stack([**x_drug1**, **x_drug2**, ...])，形状 [batch, n_drug_chunks, 128]
> **side_seq** = stack([**x_side1**, **x_side2**])，形状 [batch, n_side_chunks, 128]

**步骤2：Drug→Side交叉注意力**

药物块作为Query，去「询问」不良反应块：

- Query：**drug_seq** [batch, n_drug_chunks, 128]
- Key/Value：**side_seq** [batch, n_side_chunks, 128]

注意力计算（4头）：
1. 每个头将Q、K、V投影到32维（128/4=32）
2. 计算注意力分数：**scores** = (**Q** × **K**^T) / √32
3. Softmax归一化得到权重 **α**，形状 [batch, n_drug_chunks, n_side_chunks]
4. 加权聚合：**output** = **α** × **V**
5. 4个头拼接并投影回128维

**数学意义**：α[i][j] 表示第 i 个药物块对第 j 个ADR块的关注程度。模型学习「哪种药物特征与哪种ADR特征最相关」。

**步骤3：Side→Drug交叉注意力**

不良反应块作为Query，去「询问」药物块（流程同上，方向相反）。

**步骤4：残差连接与归一化**

> **drug_enhanced** = LayerNorm(**drug_seq** + **attn_output_drug**)
> **side_enhanced** = LayerNorm(**side_seq** + **attn_output_side**)

残差连接保留原始信息，LayerNorm稳定训练。

**最终输出**：增强后的特征块列表，每个块仍是128维，但融合了对方模态的信息。

#### 4.2.4 交互图构建

**一、交互矩阵生成（外积）**

对每对特征块计算外积（Outer Product）：

设 **drug_chunk_i** 为第 i 个药物块（128维列向量），**side_chunk_j** 为第 j 个ADR块（128维行向量）：

> **M_ij** = **drug_chunk_i** × **side_chunk_j**^T

即：128维列向量 × 128维行向量 = 128×128的矩阵。

矩阵中每个元素 M_ij[a][b] = drug_chunk_i[a] × side_chunk_j[b]，表示药物第a维特征与ADR第b维特征的交互强度。

**二、交互图堆叠**

假设有3个Drug块和2个ADR块，则生成 3×2=6 个交互矩阵。

将这6个矩阵堆叠为4D张量：

> **InteractionMap** = stack([M_00, M_01, M_10, M_11, M_20, M_21])

形状：[batch_size, 6, 128, 128]

**三、ResNet处理**

将交互图视为6通道的128×128图像，通过3个残差块处理：

**残差块结构**：

输入 → Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → (+输入) → 输出

每个残差块：
- 第一个卷积：将通道数变换（如 6→32）
- 第二个卷积：保持通道数
- 残差连接：输入直接加到输出上（需维度匹配）

经过3个残差块后，特征图尺寸逐渐缩小（通过stride=2），最终展平为128维向量：

> **h_interaction** = flatten(ResNet(**InteractionMap**))

**四、最终特征融合**

将三部分特征拼接：

> **h_total** = [**x_drug_embed**, **h_interaction**, **x_side_embed**]

形状：[batch, 128 + 128 + 128] = [batch, 384]

然后通过全连接层进行分类和回归预测。

#### 4.2.5 对比学习模块 (CCL-ASPS 2024)

对比学习通过最大化正样本对之间的相似度、最小化负样本对之间的相似度，来提升特征表征的区分能力。

---

**一、噪声视图生成**

给定融合后的特征向量 **h_total**（384维），生成带噪声的扰动视图：

**步骤1：生成随机噪声**

> **noise** = 随机采样自标准正态分布，形状与 **h_total** 相同

**步骤2：噪声归一化**

将噪声向量进行L2归一化：

> **noise_normalized** = **noise** / ||**noise**||₂

其中 ||·||₂ 表示L2范数（向量各元素平方和再开方）。

**步骤3：添加有方向的噪声**

噪声的方向与原始特征的符号一致：

> **h_noisy** = **h_total** + sign(**h_total**) × **noise_normalized** × 0.1

解释：
- sign(**h_total**) 返回每个元素的符号（+1或-1）
- 乘以归一化噪声和缩放因子0.1
- 这样噪声会在原始特征的「同方向」上做小幅扰动

---

**二、投影头（Projection Head）**

将原始特征和噪声视图特征都映射到低维空间进行对比：

**投影网络结构**：

输入（384维）→ 线性层（384→384）→ ReLU → 线性层（384→192）→ 输出

**原始特征投影**：

> **z_fused** = ProjectionHead(**h_total**)

结果：192维向量

**噪声视图投影**：

> **z_noisy** = ProjectionHead(**h_noisy**)

结果：192维向量

**L2归一化**：

将投影后的向量进行单位化：

> **z_fused** = **z_fused** / ||**z_fused**||₂
> **z_noisy** = **z_noisy** / ||**z_noisy**||₂

归一化后每个向量的L2范数为1，这样点积就等于余弦相似度。

---

**三、相似度矩阵计算**

对于一个batch（如128个样本），计算噪声视图与原始特征之间的相似度矩阵：

> **S** = (**z_noisy** × **z_fused**^T) / τ

其中：
- **z_noisy** 形状 [128, 192]
- **z_fused**^T 形状 [192, 128]
- 结果 **S** 形状 [128, 128]
- τ = 0.07 是温度参数

**矩阵解读**：
- S[i][j] 表示第 i 个样本的噪声视图与第 j 个样本原始特征的相似度
- **对角线元素** S[i][i] 是**正样本对**（同一样本的两个视图）
- **非对角线元素** S[i][j] (i≠j) 是**负样本对**（不同样本）

温度参数 τ 的作用：
- τ 越小，相似度分布越「尖锐」，模型越关注区分相似样本
- τ 越大，相似度分布越「平滑」，对差异不敏感

---

**四、InfoNCE损失计算**

目标：让每个样本的噪声视图与其原始特征最相似（对角线最大），与其他样本的原始特征不相似（非对角线最小）。

**步骤1：构建标签**

标签就是单位矩阵的对角线索引：

> labels = [0, 1, 2, ..., 127]

表示第 i 个噪声视图的正确匹配是第 i 个原始特征。

**步骤2：交叉熵损失**

将相似度矩阵的每一行视为分类logits，目标是预测正确的列索引：

> L_contrastive = CrossEntropy(**S**, labels)

展开公式：

> L_contrastive = -1/N × Σ_i log(exp(S[i][i]) / Σ_j exp(S[i][j]))

解释：
- 分子 exp(S[i][i]) 是正样本对的相似度（希望大）
- 分母 Σ_j exp(S[i][j]) 是所有样本对的相似度之和
- 最小化损失意味着最大化 S[i][i] 相对于其他 S[i][j] 的比例

---

**五、损失整合**

对比学习损失作为辅助损失，与主损失加权相加：

> L_total = λ_cls × L_classification + λ_reg × L_regression + λ_contrastive × L_contrastive

典型权重配置：
- λ_cls = 0.7（分类损失权重）
- λ_reg = 0.3（回归损失权重）
- λ_contrastive = 0.1（对比损失权重）

对比学习的效果：
- 让同一样本的不同扰动视图特征更接近
- 让不同样本的特征更分散
- 提升模型对噪声的鲁棒性和特征的判别能力

#### 4.2.6 预测头

**二分类输出：**
```
融合特征 [batch_size, embed_dim]
  ↓
fc_layer: Linear(embed_dim, embed_dim)
  ↓
BatchNorm + ReLU
  ↓
Dropout(dropout2)
  ↓
output_layer: Linear(embed_dim, 1)
  ↓
scores_one [batch_size, 1]（二分类logits）
```

**回归输出：**
```
融合特征 [batch_size, embed_dim]
  ↓
fc_layer: Linear(embed_dim, embed_dim)
  ↓
BatchNorm + ReLU
  ↓
Dropout(dropout2)
  ↓
output_layer: Linear(embed_dim, 1)
  ↓
scores_two [batch_size, 1]（回归值）
```

---

## 5. 损失函数

### 5.1 二分类损失

**稀疏多标签分类交叉熵（Sparse Multi-label Categorical Cross-entropy）：**

```python
def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """
    Args:
        y_true: 真实标签 [batch_size]，值为0或1
        y_pred: 预测logits [batch_size]
        
    Returns:
        loss: 标量损失值
    """
    # 步骤1：调整预测值
    y_pred = (1 - 2 * y_true) * y_pred
    
    # 步骤2：分离正负样本预测
    y_pred_neg = y_pred - y_true * 1e12      # 负样本预测（正样本设为-∞）
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # 正样本预测（负样本设为-∞）
    
    # 步骤3：添加零向量用于logsumexp
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    # 步骤4：计算损失
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)  # log(1 + exp(neg_pred))
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)  # log(1 + exp(pos_pred))
    
    return neg_loss + pos_loss
```

**等价形式（使用BCEWithLogitsLoss）：**
```python
loss_cls = BCEWithLogitsLoss()(scores_one, labels)
```

### 5.2 回归损失

**均方误差损失（MSE Loss）：**
```python
loss_reg = MSELoss()(scores_two[positive_indices], ratings[positive_indices])
```

**说明：** 仅在正样本上计算回归损失（即标签为1的样本）

### 5.3 总损失

**加权组合：**
```python
lambda_cls = 0.7  # 二分类损失权重
lambda_reg = 0.3  # 回归损失权重

total_loss = lambda_cls * loss_cls + lambda_reg * loss_reg
```

**对比学习损失（可选）：**
```python
if use_contrastive_learning:
    loss_contrastive = contrastive_module(features_list, fused_features)
    total_loss += contrastive_weight * loss_contrastive
```

---

## 6. 训练及测试

### 6.1 训练流程

#### 6.1.1 5折交叉验证

```
数据集划分：
├── 总样本数：len(final_positive_sample) + len(final_negative_sample)
├── 5折划分：使用StratifiedKFold
│   ├── 保持正负样本比例
│   ├── 随机状态：random_state=5
│   └── 打乱数据：shuffle=True
└── 对每一折：
    ├── 训练集：80%
    ├── 测试集：20%
    └── 验证集：从训练集中分离
```

#### 6.1.2 单个Epoch的训练

```
train(model, train_loader, device, global_drug_features, 
      global_side_features, optimizer, lossfunction1, lossfunction2, 
      graph_builder=None)

步骤：
1. 模型设置为训练模式：model.train()

2. 对每个batch：
   a. 提取batch数据：(drug_idx, side_idx, ratings)
   
   b. 构建二分类标签：labels = (ratings > 0).float()
   
   c. 前向传播：
      scores_one, scores_two = model(
          drug_indices=drug_idx,
          side_indices=side_idx,
          device=device,
          global_drug_features=global_drug_features,
          global_side_features=global_side_features
      )
   
   d. 计算损失：
      loss_cls = lossfunction1(scores_one, labels)  # BCEWithLogitsLoss
      loss_reg = lossfunction2(scores_two[positive_mask], 
                               ratings[positive_mask])  # MSELoss
      total_loss = 0.7 * loss_cls + 0.3 * loss_reg
   
   e. 反向传播：
      optimizer.zero_grad()
      total_loss.backward(retain_graph=True)
   
   f. 梯度裁剪（可选）：
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
   
   g. 参数更新：
      optimizer.step()

3. 返回平均损失
```

#### 6.1.3 学习率调度

**ReduceLROnPlateau：**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',           # 最大化验证AUC
    factor=0.5,           # 学习率乘以0.5
    patience=10,          # 10个epoch无改进后降低学习率
    min_lr=1e-6
)

# 在每个epoch后调用
scheduler.step(val_auc)
```

#### 6.1.4 超参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 100 | 训练轮数 |
| `lr` | 1e-3 | 初始学习率 |
| `embed_dim` | 128 | 嵌入维度 |
| `weight_decay` | 1e-5 | L2正则化系数 |
| `batch_size` | 128 | 训练批大小 |
| `dropout1` | 0.3 | HAN编码器dropout率 |
| `dropout2` | 0.3 | 预测头dropout率 |
| `label_smooth` | 0.05 | 标签平滑系数（0~0.2） |
| `grad_clip` | 0.5 | 梯度裁剪阈值 |
| `han_layers` | 3 | HAN层数 |
| `han_heads` | 4 | HAN注意力头数 |
| `subgraph_k_hop` | 3 | 子图采样K-hop跳数 |
| `subgraph_max_nodes` | 300 | 子图最大节点数 |
| `contrastive_weight` | 0.1 | 对比学习损失权重 |

### 6.2 测试流程

```
test(model, test_loader, device, global_drug_features, 
     global_side_features, lossfunction1, lossfunction2, 
     graph_builder=None)

步骤：
1. 模型设置为评估模式：model.eval()

2. 对每个batch：
   a. 提取batch数据：(drug_idx, side_idx, ratings)
   
   b. 构建二分类标签：labels = (ratings > 0).float()
   
   c. 前向传播（无梯度）：
      with torch.no_grad():
          scores_one, scores_two = model(...)
   
   d. 计算损失（同训练）
   
   e. 收集预测结果：
      prob_one = sigmoid(scores_one)  # 二分类概率
      pred1.append(prob_one)
      pred2.append(scores_two)        # 回归值
      ground_truth.append(ratings)
      label_truth.append(labels)

3. 计算评估指标：
   a. 二分类指标：
      - AUC: roc_auc_score(label_truth, pred1)
      - PR-AUC: auc(precision_recall_curve)
      - Accuracy: accuracy_score(label_truth, pred1 >= 0.5)
      - MCC: matthews_corrcoef(label_truth, pred1 >= 0.5)
   
   b. 回归指标（仅在正样本上）：
      - RMSE: sqrt(MSE(pred2[positive], ground_truth[positive]))
      - MAE: mean_absolute_error(pred2[positive], ground_truth[positive])

4. 返回所有指标和预测结果
```

### 6.3 评估指标

| 指标 | 类型 | 范围 | 说明 |
|------|------|------|------|
| **AUC** | 二分类 | [0, 1] | ROC曲线下面积，越高越好 |
| **PR-AUC** | 二分类 | [0, 1] | 精准率-召回率曲线下面积 |
| **Accuracy** | 二分类 | [0, 1] | 分类准确率（阈值0.5） |
| **MCC** | 二分类 | [-1, 1] | Matthews相关系数，考虑不平衡 |
| **RMSE** | 回归 | [0, ∞) | 均方根误差，越低越好 |
| **MAE** | 回归 | [0, ∞) | 平均绝对误差，越低越好 |

### 6.4 完整训练流程

```python
# 1. 数据加载和预处理
drug_list, adr_list, drug_side = load_label(...)
addition_neg, pos_samples, neg_samples = Extract_positive_negative_samples(...)
final_samples = vstack([pos_samples, neg_samples])

# 2. 特征加载
drug_features = load_drug_feature(...)
adr_features = load_adr_feature(...)

# 3. 5折交叉验证
kfold = StratifiedKFold(5, random_state=5, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(kfold.split(...)):
    
    # 4. 数据集划分
    train_data = data[train_idx]
    test_data = data[test_idx]
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    # 5. 模型初始化
    model = DGAPred(...)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10)
    
    # 6. 训练循环
    for epoch in range(100):
        train_loss, _ = train(model, train_loader, ...)
        
        # 7. 验证
        val_auc, val_pr_auc, val_rmse, val_mae, val_acc, val_mcc, \
            _, _, _, _, _, _, _ = test(model, test_loader, ...)
        
        # 8. 学习率调度
        scheduler.step(val_auc)
        
        # 9. 早停（可选）
        if no_improvement_for_20_epochs:
            break
    
    # 10. 测试
    test_auc, test_pr_auc, test_rmse, test_mae, test_acc, test_mcc, \
        _, _, _, _, _, _, _ = test(model, test_loader, ...)
    
    # 11. 保存结果
    results.append({
        'fold': fold,
        'auc': test_auc,
        'pr_auc': test_pr_auc,
        'rmse': test_rmse,
        'mae': test_mae,
        'acc': test_acc,
        'mcc': test_mcc
    })

# 12. 统计5折结果
mean_auc = mean(all_aucs)
std_auc = std(all_aucs)
# ... 其他指标
```

### 6.5 输出结果

**输出目录结构：**
```
output_{timestamp}/
├── fold_1/
│   ├── model_best.pth          # 最佳模型权重
│   ├── predictions.csv         # 预测结果
│   └── metrics.json            # 评估指标
├── fold_2/
│   └── ...
├── fold_3/
│   └── ...
├── fold_4/
│   └── ...
├── fold_5/
│   └── ...
└── summary.csv                 # 5折平均结果
```

**结果汇总：**
```
Fold 1: AUC=0.85, PR-AUC=0.82, RMSE=0.25, MAE=0.18, Acc=0.81, MCC=0.62
Fold 2: AUC=0.84, PR-AUC=0.81, RMSE=0.26, MAE=0.19, Acc=0.80, MCC=0.60
Fold 3: AUC=0.86, PR-AUC=0.83, RMSE=0.24, MAE=0.17, Acc=0.82, MCC=0.64
Fold 4: AUC=0.83, PR-AUC=0.80, RMSE=0.27, MAE=0.20, Acc=0.79, MCC=0.58
Fold 5: AUC=0.85, PR-AUC=0.82, RMSE=0.25, MAE=0.18, Acc=0.81, MCC=0.62

Mean: AUC=0.847±0.011, PR-AUC=0.816±0.011, RMSE=0.254±0.011, 
      MAE=0.184±0.011, Acc=0.806±0.011, MCC=0.612±0.022
```

---

## 7. 关键技术创新

### 7.1 异质图神经网络 (HAN)

- **多元路径学习**：同时学习Drug-Gene-ADR的多条传播路径
- **节点级注意力**：在每条元路径上独立学习注意力权重
- **语义级注意力**：融合Gene节点从不同入边类型接收的信息
- **性能优化**：
  - 移除全局注意力（O(n²)复杂度）
  - 使用torch_scatter.scatter_add替代循环聚合
  - 支持多尺度注意力（local + medium）

### 7.2 特征交互注意力 (FIA-DTA 2025)

- **多模态特征融合**：药物和不良反应的多个特征维度交互
- **双向注意力**：药物块→不良反应块 和 不良反应块→药物块
- **特征分块**：将高维特征分解为多个低维块，提高计算效率

### 7.3 对比学习 (CCL-ASPS 2024)

- **多视图学习**：利用多个特征视图进行对比学习
- **InfoNCE损失**：最大化正样本对相似度，最小化负样本对相似度
- **表征增强**：提高模型学到的特征表征质量

### 7.4 K-hop自适应采样

- **自适应邻居采样**：根据节点重要性（度中心性）动态调整采样数量
- **大小限制**：防止子图过大导致内存溢出
- **种子节点优先**：确保所有batch节点都被保留

---

## 8. 依赖库

```
torch>=1.9.0
torch-geometric>=2.0.0
torch-scatter>=2.0.0
scikit-learn>=0.24.0
pandas>=1.1.0
numpy>=1.19.0
scipy>=1.5.0
tqdm>=4.50.0
```

---

## 9. 使用示例

### 9.1 训练模型

```bash
python src/main.py \
    --epochs 100 \
    --lr 1e-3 \
    --embed_dim 128 \
    --batch_size 128 \
    --dropout1 0.3 \
    --dropout2 0.3 \
    --use_heterogeneous_gnn \
    --use_feature_interaction \
    --use_contrastive_learning \
    --han_layers 3 \
    --han_heads 4
```

### 9.2 自定义配置

```python
import argparse
from src.main import train_test

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--embed_dim', type=int, default=128)
# ... 其他参数

args = parser.parse_args()
train_test(drug_features, adr_features, train_data, test_data, 
           fold, args, drug_list, adr_list, output_dir)
```

---

## 10. 性能优化建议

1. **GPU内存优化**：
   - 减小子图最大节点数（`subgraph_max_nodes`）
   - 降低批大小（`batch_size`）
   - 使用混合精度训练（`torch.cuda.amp`）

2. **训练加速**：
   - 启用多尺度注意力（`use_multi_scale_attention`）
   - 增加HAN层数（`han_layers`）
   - 使用梯度累积

3. **模型泛化**：
   - 调整dropout率（`dropout1`, `dropout2`）
   - 启用标签平滑（`label_smooth`）
   - 增加对比学习权重（`contrastive_weight`）

---

## 11. 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| OOM错误 | GPU内存不足 | 减小batch_size或subgraph_max_nodes |
| 损失NaN | 梯度爆炸 | 启用梯度裁剪或降低学习率 |
| 性能不佳 | 超参数不合适 | 进行网格搜索或随机搜索 |
| 训练缓慢 | 子图过大 | 减小k_hop或max_neighbors_per_node |

---

**文档版本**：1.0  
**最后更新**：2025年  
**维护者**：DGAPred Team
