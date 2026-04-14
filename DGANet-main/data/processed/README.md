# 处理后的 CTD 数据集说明

## 1. 数据简介

本文件夹包含经过清洗和筛选的 CSV 文件，源自比较毒理基因组学数据库 (Comparative Toxicogenomics Database, CTD)。这些数据经过处理，只保留了具有统计学显著性 (Corrected P-Value < 0.05) 的关联，主要用于描述化学物质（药物）和疾病的生物学功能特性。

### 文件列表
1.  **`CTD_chem_go_filtered_with_pertid.csv`**: 化学物质与基因本体 (GO) 术语的关联数据（已映射到LINCS pert_id）。
2.  **`CTD_chem_pathways_filtered_with_pertid.csv`**: 化学物质与生物通路 (Pathway) 的关联数据（已映射到LINCS pert_id）。
3.  **`CTD_diseases_pathways_filtered.csv`**: 疾病与生物通路 (Pathway) 的关联数据。

---

## 2. 数据处理流程

### 2.1 药物-GO/Pathway 数据处理流程

整个处理流程分为三个阶段，由 `pert2cid/` 目录下的脚本完成：

#### 阶段一：P值筛选与SMILES获取 (`pubchem_get.py`)

**富集分析与P值的关系**：

CTD数据库中的药物-GO/Pathway关联来自**富集分析 (Enrichment Analysis)**。富集分析的核心思想是：
- 某种药物会影响一组基因的表达
- 如果这组基因中，属于某个GO术语或Pathway的基因数量**显著高于随机预期**，则认为该药物与该GO/Pathway存在富集关联

**P值的含义**：
- P值表示"观察到的富集程度纯属偶然"的概率
- 例如：P=0.01 意味着只有1%的可能性是随机巧合
- **CorrectedPValue** 是经过多重检验校正（如Bonferroni或FDR校正）后的P值，比原始P值更保守可靠

**为什么选择 P < 0.05**：
- 0.05是生物医学领域广泛接受的统计显著性阈值
- P < 0.05 表示该关联有95%以上的置信度是真实的，而非随机噪声
- 保留P值过高的关联会引入大量假阳性，干扰模型学习真正的生物学规律
- 这是**数据质量控制**的关键步骤，确保输入模型的都是可靠的药物-功能关联

**处理步骤**：
1. 读取原始CTD数据 (`CTD_chem_go_enriched1.csv`, `CTD_chem_pathways_enriched1.csv`)
2. **P值筛选**：只保留 `CorrectedPValue < 0.05` 的统计显著关联
3. **取交集**：只保留同时出现在GO和Pathway数据中的化学物质（确保数据一致性）
4. **获取SMILES**：通过PubChem REST API，根据化学名查询其标准SMILES结构式
5. 输出中间文件：`CTD_chem_go_enriched_withsmiles_pubchem.csv`, `CTD_chem_pathways_enriched_withsmiles_pubchem.csv`

**设计理由**：
- **P值筛选**：确保只使用统计学上可靠的药物-功能关联，避免噪声数据干扰模型
- **获取SMILES**：CTD原始数据使用MeSH ID标识化学物质，而LINCS数据库使用pert_id；SMILES作为分子结构的通用表示，可作为两个数据库之间的桥梁

#### 阶段二：SMILES到pert_id的映射 (`process_cid_pertid.py`)

**处理步骤**：
1. 读取带SMILES的CTD数据和LINCS药物列表 (`drug_pert_similes_list.csv`)
2. **直接匹配**：首先尝试SMILES字符串完全相同的直接匹配
3. **MACCS指纹相似度匹配**：对未直接匹配的数据，使用MACCS分子指纹计算Tanimoto相似度
   - 预计算所有LINCS药物的MACCS指纹（避免重复计算）
   - 对每个未匹配的SMILES，找到相似度最高且≥阈值的pert_id
4. **相似度阈值筛选**：默认阈值为1.0（即只保留完全匹配），可调整为0.75-0.85以允许近似匹配
5. 输出最终文件：`CTD_chem_go_filtered_with_pertid.csv`, `CTD_chem_pathways_filtered_with_pertid.csv`

**设计理由**：
- **两阶段匹配策略**：直接匹配速度快且准确，相似度匹配作为补充可以找到结构相似但SMILES表示略有差异的分子
- **MACCS指纹**：166位的分子结构键指纹，计算效率高，适合大规模相似度筛选
- **Tanimoto相似度**：化学信息学中最常用的分子相似度度量，取值范围[0,1]
- **可调阈值**：允许根据数据质量和匹配需求灵活调整，平衡覆盖率和准确性

### 2.2 疾病-Pathway 数据处理流程 (`diseases_process.py`)

**处理步骤**：
1. 读取SIDER数据集中的有效MESH_ID集合 (`sider_pert_mesh_list.csv`)
2. 读取CTD疾病-通路关联数据 (`CTD_diseases_pathways.csv`)
3. **筛选**：只保留DiseaseID在SIDER有效MESH_ID集合中的记录
4. 提取 `DiseaseID` 和 `PathwayID` 列并去重
5. 输出：`CTD_diseases_pathways_filtered.csv`

**设计理由**：
- **与SIDER对齐**：SIDER数据集包含药物-副作用关联，其中副作用以MESH_ID标识；只保留SIDER中存在的疾病ID，确保疾病-通路数据可以与药物-副作用数据关联
- **构建完整知识图谱**：通过Pathway作为桥梁，可以建立 Drug → Pathway ← Disease 的关联路径，为模型提供药物-疾病的间接关联信息

---

## 3. 数据关系详解

### 3.1 `CTD_chem_go_filtered_with_pertid.csv` (药物-GO 关联)
该文件展示了 **药物 (pert_id)** 与 **基因本体术语 (GOTermID)** 之间的富集关系。

*   **关系含义**: 表示某种药物显著地影响了特定的生物学功能。GO 术语涵盖三个方面：
    *   **生物过程 (Biological Process)**: 如 "细胞凋亡"、"免疫反应"。
    *   **分子功能 (Molecular Function)**: 如 "酶活性"、"受体结合"。
    *   **细胞组分 (Cellular Component)**: 如 "线粒体"、"细胞核"。
*   **列说明**:
    *   `pert_id`: LINCS药物扰动标识符。
    *   `GOTermID`: 基因本体术语的唯一标识符。

### 3.2 `CTD_chem_pathways_filtered_with_pertid.csv` (药物-通路 关联)
该文件展示了 **药物 (pert_id)** 与 **生物通路 (PathwayID)** 之间的富集关系。

*   **关系含义**: 表示某种药物显著地参与或影响了特定的代谢或信号通路。
*   **数据来源**: KEGG 和 REACTOME 等通路数据库。
*   **列说明**:
    *   `pert_id`: LINCS药物扰动标识符。
    *   `PathwayID`: 生物通路的唯一标识符（如 KEGG:hsa05200）。

### 3.3 `CTD_diseases_pathways_filtered.csv` (疾病-通路 关联)
该文件展示了 **疾病 (DiseaseID)** 与 **生物通路 (PathwayID)** 之间的关联。

*   **关系含义**: 表示某种疾病与特定生物通路存在关联（疾病可能由该通路异常引起，或该通路是疾病的治疗靶点）。
*   **列说明**:
    *   `DiseaseID`: 疾病的MESH ID标识符。
    *   `PathwayID`: 生物通路的唯一标识符。

---

## 4. 为什么有助于预测？

这些数据集为药物预测模型（如药物-药物相互作用预测、副作用预测、适应症预测等）提供了关键的 **生物学特征 (Biological Features)**。

### 4.1 揭示潜在机制 (Mechanism of Action)
单纯的化学结构信息（如 SMILES）虽然能反映分子特性，但很难直接推断其在生物体内的复杂反应。
*   **GO 数据** 告诉我们药物在微观层面"做了什么"（如结合了什么受体，干扰了什么过程）。
*   **Pathway 数据** 告诉我们药物在宏观层面"影响了哪条路"（如阻断了癌症信号通路）。
这相当于给模型提供了药物的 **"功能指纹"**。

### 4.2 基于"功能相似性"的推断
**基本假设**: 具有相似生物学功能谱（相似的 GO/Pathway 特征）的药物，往往具有相似的：
*   **治疗效果**: 可能治疗同一种疾病。
*   **副作用 (ADR)**: 可能引起类似的毒性反应。
*   **相互作用**: 可能与其他药物发生类似的冲突。

**示例**:
如果药物 A 和药物 B 都能富集到 "细胞色素 P450 代谢通路"，那么模型就可以推断它们如果联用可能会发生代谢竞争，从而预测出潜在的药物相互作用 (DDI)。

### 4.3 疾病-通路关联的价值
通过引入疾病-通路数据，模型可以：
*   **建立药物-疾病的间接关联**：Drug → Pathway ← Disease，即如果药物影响某通路，而该通路与某疾病相关，则药物可能对该疾病有治疗或副作用潜力
*   **理解副作用机制**：某些副作用本质上是疾病状态，通过通路关联可以解释为什么某药物会引起特定副作用

### 4.4 提高模型的泛化能力
通过引入 GO、Pathway 和疾病信息，模型不再仅仅依赖于化学结构的相似性，而是可以利用更深层的生物学语义信息。这对于结构差异大但功能相似的药物（骨架跃迁）特别有用.
