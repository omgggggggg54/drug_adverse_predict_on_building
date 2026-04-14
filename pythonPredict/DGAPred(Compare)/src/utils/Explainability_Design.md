# DGAPred 模型可解释性模块设计与实现说明书

## 1. 模块概述

本文档旨在解释 `src/utils/explainability.py` 模块的设计原理与实现细节。该模块为 DGAPred 模型（基于异质图神经网络 HAN）提供预测结果的可解释性分析。

鉴于深度学习模型（尤其是图神经网络）的“黑盒”特性，单纯依靠 Attention 权重往往只能反映局部的节点交互，难以量化基因节点对最终预测结果（Drug-ADR 关联评分）的全局贡献。为此，本模块选用了 **积分梯度 (Integrated Gradients, IG)** 算法，通过计算基因 Embedding 在模型预测路径上的梯度积分，来实现对基因重要性的精确归因。

## 2. 核心算法：积分梯度 (Integrated Gradients)

积分梯度是一种满足“完备性（Completeness）”公理的归因方法，由 Sundararajan 等人在 ICML 2017 提出。它有效解决了传统梯度法（Saliency Maps）可以通过梯度饱和导致重要特征归因为零的问题。

### 核心思想
如果我们将输入特征从“基线状态”（Baseline，如零向量）连续变化到“实际输入状态”，那么在这个过程中，模型输出相对于输入的梯度的积分，就等于该输入特征对预测结果的总贡献。

### 计算公式
$$IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha$$

为了在计算机中实现，我们使用黎曼和（Riemann Sum）进行离散化近似：

$$IG_i(x) \approx (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m} \times (x - x'))}{\partial x_i}$$

其中：
*   $x$：实际输入（训练好的 Gene Embedding 向量）
*   $x'$：基线输入（全零向量，代表该基因不存在或被屏蔽）
*   $\alpha$：插值系数，从 0 到 1 线性变化
*   $m$：积分步数（`n_steps`，默认 50）

## 3. 代码实现详解

该功能封装在 `IntegratedGradientExplainer` 类中，具体实现路径如下：

### 3.1 基线定义 (Baseline)
代码中隐式采用了 **全零向量 (Zero Vector)** 作为基线。
*   **依据**：在 Embedding 向量空间中，零向量通常意味着“零激活”或“无信息”，是作为基因节点“缺席”对照组的理想选择。

### 3.2 路径积分实现 (Path Integration)
对应方法：`explain` 和 `_compute_gradients_at_scale`

为了计算从基线到实际输入的积分，程序并没有修改输入数据 Tensor 本身，而是采用了更高效的 **权重缩放** 策略：

1.  **线性插值步骤**：
    将 `0` 到 `1` 的区间分为 `n_steps`（默认 50）个步长。
    在第 `k` 步，计算缩放因子 `scale = k / n_steps`。

2.  **缩放 Embedding 权重**：
    在 `_forward_with_scaled_embedding` 方法中，代码动态修改了模型的 Embedding 层参数：
    ```python
    # 临时将 Embedding 权重缩放 scale 倍
    # 物理含义：模拟当前所有基因的特征强度仅为正常水平的 scale 比例
    scaled_weight = original_weight * scale
    self.model.gene_embedding.weight = nn.Parameter(scaled_weight)
    ```
    **优势**：这种方法无需侵入修改模型内部复杂的图卷积（HAN）和子图采样逻辑，直接从源头控制了基因特征的强度，保证了计算图的完整性。

3.  **梯度计算**：
    在每次前向传播得到预测值（Logits）后，执行 `backward()`。
    利用 PyTorch 的自动微分机制，获取 `self.model.gene_embedding.weight.grad`。
    *   该梯度代表了：在当前特征强度下，基因 Embedding 的微小扰动对预测结果的影响率。

4.  **积分近似与累积**：
    将这 `n_steps` 次计算得到的梯度进行累加，最后除以步数 `n_steps` 得到平均梯度。

### 3.3 归因聚合 (Attribution Aggregation)
由于每个基因由一个 128 维（`embed_dim`）的向量表示，积分梯度计算出的归因结果也是一个同维度的向量。为了得到一个标量的“重要性分数”，代码进行了如下处理：

```python
# 1. 计算近似积分：平均梯度 * (实际值 - 基线值)
# 由于基线是 0，即：integrated_grads = avg_grads * actual_embeddings
integrated_grads = (integrated_grads / self.n_steps) * actual_embeddings

# 2. 向量归约：计算 L2 范数
gene_importance = torch.norm(integrated_grads, p=2, dim=1)
```

**物理含义**：计算基因 Embedding 向量在预测方向上的投影长度（贡献强度）。

## 4. 特殊处理与鲁棒性设计

*   **图结构的稀疏性处理**：
    代码中包含 `if grad is None` 的检查。这是因为 DGAPred 使用了子图采样（Subgraph Sampling）策略。对于特定的 Drug-ADR 对，只有位于其 K-hop 邻域内的基因才会被采样并参与计算。
    *   **处理逻辑**：未参与前向传播的基因，其梯度自然为 `None`。代码将其贡献正确置为 `0`，确保了程序的稳健运行。

*   **计算图重置**：
    在每次计算梯度后，代码会立即恢复原始的 Embedding 权重 `self.model.gene_embedding.weight = nn.Parameter(original_weight)`，防止影响后续的推理或训练。

## 5. 总结
该模块通过积分梯度算法，实现了一种 **模型无关 (Model-Agnostic)** 且 **公理化 (Axiomatic)** 的可解释性分析。它不仅能捕捉基因自身的特征贡献，还能隐式地包含该基因通过图神经网络的多跳消息传递机制对预测结果产生的间接影响，是解释 DGAPred 这类复杂图模型的有效手段。
