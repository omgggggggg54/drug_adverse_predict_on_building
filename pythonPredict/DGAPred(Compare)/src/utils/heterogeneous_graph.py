"""Heterogeneous Graph Construction and Sampling Module

This module provides utilities for building heterogeneous graphs from raw data
and sampling K-hop subgraphs for batch processing.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import torch
from tqdm import tqdm


class HeterogeneousGraphBuilder:
    """Build heterogeneous graph from raw data files.
    
    Node types: Drug (pert_id), Gene (GeneSymbol), ADR (MESH_ID)
    Edge types: Drug-Gene, Gene-ADR (bidirectional)
    """
    
    def __init__(self, rawpath: str, drug_list: List[str], adr_list: List[str]):
        """Initialize graph builder.
        
        Args:
            rawpath: Path to raw data directory
            drug_list: List of drug IDs (pert_id)顺序与全局是对上的
            adr_list: List of ADR IDs (MESH_ID)顺序与全局是对上的
        """
        self.rawpath = rawpath
        self.drug_list = drug_list
        self.adr_list = adr_list
        
        # Node ID mappings: {node_type: {original_id: node_idx}}
        self.drug_id_to_idx: Dict[str, int] = {drug_id: idx for idx, drug_id in enumerate(drug_list)}
        self.adr_id_to_idx: Dict[str, int] = {adr_id: idx for idx, adr_id in enumerate(adr_list)}
        self.gene_id_to_idx: Dict[str, int] = {}
        
        # Reverse mappings: {node_type: {node_idx: original_id}}
        self.drug_idx_to_id: Dict[int, str] = {idx: drug_id for drug_id, idx in self.drug_id_to_idx.items()}
        self.adr_idx_to_id: Dict[int, str] = {idx: adr_id for adr_id, idx in self.adr_id_to_idx.items()}
        self.gene_idx_to_id: Dict[int, str] = {}
        
        # Node type offsets for unified node indexing
        self.drug_offset = 0
        self.gene_offset = len(drug_list)
        self.adr_offset = None  # Will be set after gene nodes are added
        
        # Edge lists: {edge_type: [(src_idx, dst_idx), ...]}
        # 只保留drug_gene和gene_adr边，移除drug_adr直接边
        # 添加反向边支持双向传播
        self.edges: Dict[str, List[Tuple[int, int]]] = {
            'drug_gene': [],      # Drug -> Gene (正向)
            'gene_adr': [],       # Gene -> ADR (正向)
            'gene_drug': [],      # Gene -> Drug (反向，用于adr->gene->drug传播)
            'adr_gene': []        # ADR -> Gene (反向，用于adr->gene->drug传播)
        }
        
        # Build graph
        self._build_graph()
    
    def _build_graph(self):
        """Build heterogeneous graph from raw data files."""
        print(f"\n{'='*60}")
        print("Building Heterogeneous Graph")
        print(f"{'='*60}")
        
        # Step 1: Build Drug-Gene edges and collect unique genes
        print("Loading Drug-Gene interactions...")
        pd_DGen = pd.read_csv(self.rawpath + "ctd_chem_pert_gene_ixns_list.csv", 
                              header=0, delimiter='\t')
        
        # Filter to drugs in our list
        pd_DGen = pd_DGen[pd_DGen['pert_id'].isin(self.drug_list)]
        pd_DGen = pd_DGen.drop_duplicates(subset=['pert_id', 'GeneSymbol'], keep='first')
        
        # Collect unique genes
        unique_genes = sorted(pd_DGen['GeneSymbol'].unique().tolist())
        self.gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(unique_genes)}
        self.gene_idx_to_id = {idx: gene_id for gene_id, idx in self.gene_id_to_idx.items()}
        self.adr_offset = self.gene_offset + len(unique_genes)#adr节点的偏移量,前面是基因节点
        
        print(f"  - Drugs: {len(self.drug_list)}")
        print(f"  - Genes: {len(unique_genes)}")
        print(f"  - ADRs: {len(self.adr_list)}")
        
        # Build Drug-Gene edges (使用 itertuples() 加速)
        for row in tqdm(pd_DGen.itertuples(index=False), 
                       total=len(pd_DGen), 
                       desc="Building Drug-Gene edges"):
            drug_idx = self.drug_id_to_idx[row.pert_id]
            gene_idx = self.gene_offset + self.gene_id_to_idx[row.GeneSymbol]
            self.edges['drug_gene'].append((drug_idx, gene_idx))
        
        print(f"  - Drug-Gene edges: {len(self.edges['drug_gene'])}")
        
        # Step 2: Build Gene-ADR edges
        print("Loading Gene-ADR associations...")
        pd_AGen = pd.read_csv(self.rawpath + "ctd_gene_adr_asso_list_4386.csv", 
                             header=0, delimiter='\t')
        
        # Filter to ADRs in our list
        pd_AGen = pd_AGen[pd_AGen['MESH_ID'].isin(self.adr_list)]
        pd_AGen = pd_AGen.drop_duplicates(subset=['MESH_ID', 'GeneSymbol'], keep='first')
        
        # Only keep genes we've seen in Drug-Gene interactions
        pd_AGen = pd_AGen[pd_AGen['GeneSymbol'].isin(unique_genes)]
        
        # Build Gene-ADR edges (使用 itertuples() 加速)
        for row in tqdm(pd_AGen.itertuples(index=False), 
                       total=len(pd_AGen), 
                       desc="Building Gene-ADR edges"):
            gene_idx = self.gene_offset + self.gene_id_to_idx[row.GeneSymbol]
            adr_idx = self.adr_offset + self.adr_id_to_idx[row.MESH_ID]
            self.edges['gene_adr'].append((gene_idx, adr_idx))
        
        print(f"  - Gene-ADR edges: {len(self.edges['gene_adr'])}")
        
        # Step 3: Build reverse edges for bidirectional propagation
        # Gene -> Drug (反向边，用于adr->gene->drug传播)
        print("Building reverse edges...")
        for drug_idx, gene_idx in tqdm(self.edges['drug_gene'], desc="Building Gene-Drug reverse edges"):
            self.edges['gene_drug'].append((gene_idx, drug_idx))
        
        # ADR -> Gene (反向边，用于adr->gene->drug传播)
        for gene_idx, adr_idx in tqdm(self.edges['gene_adr'], desc="Building ADR-Gene reverse edges"):
            self.edges['adr_gene'].append((adr_idx, gene_idx))
        
        print(f"  - Gene-Drug reverse edges: {len(self.edges['gene_drug'])}")
        print(f"  - ADR-Gene reverse edges: {len(self.edges['adr_gene'])}")
        
        # 注意：已移除Drug-ADR直接边，信息只能通过Gene传递
        
        # Convert to sparse adjacency matrices for efficient querying
        self._build_adjacency_matrices()
        
        print(f"{'='*60}\n")
    
    def _build_adjacency_matrices(self):
        """Build combined sparse adjacency matrix for neighbor queries."""
        total_nodes = self.adr_offset + len(self.adr_list)
        shape = (total_nodes, total_nodes)
        
        def build_adj(edge_list):
            if edge_list:
                src, dst = zip(*edge_list)
                return sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=shape)
            return sp.csr_matrix(shape)
        
        # 直接构建合并后的邻接矩阵（包含4种边类型）
        adj_combined = (build_adj(self.edges['drug_gene']) + 
                       build_adj(self.edges['gene_adr']) +
                       build_adj(self.edges['gene_drug']) + 
                       build_adj(self.edges['adr_gene']))
        self.adj_combined = (adj_combined > 0).astype(np.float32)
        
    def get_node_type(self, node_idx: int) -> str:
        """Get node type from global node index."""
        if node_idx < self.gene_offset:
            return 'drug'
        elif node_idx < self.adr_offset:
            return 'gene'
        else:
            return 'adr'
    
    def get_neighbors(self, node_idx: int) -> np.ndarray:
        """Get neighbors of a node.
        
        Args:
            node_idx: Global node index
            
        Returns:
            Array of neighbor node indices
        """
        return self.adj_combined[node_idx].indices
    
    def get_total_nodes(self) -> int:
        """Get total number of nodes in the graph."""
        return self.adr_offset + len(self.adr_list)


class KHopSubgraphSampler:
    """Sample K-hop subgraphs from batch nodes with adaptive sampling.
    
    [NEW] 支持分层采样策略 (Hierarchical Layer Sampling)
    - 不同hop层使用不同的采样率
    - 第1层(直接邻居): 高采样率，保留局部结构
    - 第2层(间接邻居): 低采样率，捕获全局模式
    """
    
    def __init__(self, graph_builder: HeterogeneousGraphBuilder, 
                 k_hop: int = 2, max_nodes: int = 500, max_neighbors_per_node: int = 50,
                 use_adaptive_sampling: bool = True,
                 use_hierarchical_sampling: bool = True,
                 layer_sample_rates: List[float] = None):
        """Initialize subgraph sampler.
        
        Args:
            graph_builder: HeterogeneousGraphBuilder instance
            k_hop: Number of hops to sample
            max_nodes: Maximum number of nodes in subgraph
            max_neighbors_per_node: Maximum neighbors to sample per node (防止节点爆炸)
            use_adaptive_sampling: Whether to use adaptive neighbor sampling based on node importance
            use_hierarchical_sampling: [NEW] 是否使用分层采样策略
            layer_sample_rates: [NEW] 各层采样率，如 [0.8, 0.4] 表示第1层80%，第2层40%
        """
        self.graph_builder = graph_builder
        self.k_hop = k_hop
        self.max_nodes = max_nodes
        self.max_neighbors_per_node = max_neighbors_per_node
        self.use_adaptive_sampling = use_adaptive_sampling
        self.use_hierarchical_sampling = use_hierarchical_sampling
        
        # [NEW] 分层采样率配置
        # 默认: 第1层80%, 第2层40%, 第3层20%... (递减)
        if layer_sample_rates is None:
            self.layer_sample_rates = [0.8 / (2 ** i) for i in range(k_hop)]
        else:
            self.layer_sample_rates = layer_sample_rates
        
        # Compute node importance for adaptive sampling
        if self.use_adaptive_sampling:
            self._compute_node_importance()
    
    def _compute_node_importance(self):
        """Compute node importance scores based on degree centrality."""
        total_nodes = self.graph_builder.get_total_nodes()
        self.node_importance = np.zeros(total_nodes)
        
        # Compute degree for each node
        for node_idx in range(total_nodes):
            neighbors = self.graph_builder.get_neighbors(node_idx)
            self.node_importance[node_idx] = len(neighbors)
        
        # Normalize to [0, 1]
        max_degree = self.node_importance.max()
        if max_degree > 0:
            self.node_importance = self.node_importance / max_degree
        else:
            self.node_importance = np.ones(total_nodes)
        
        print(f"[Adaptive Sampling] Node importance computed (avg: {self.node_importance.mean():.3f})")
    
    def _adaptive_sample_neighbors(self, node_idx: int, neighbors: np.ndarray, 
                                     hop_level: int = 0) -> list:
        """Adaptively sample neighbors based on node importance.
        
        
        两种采样策略协同工作：
        1. 分层采样：根据hop层级调整最大邻居数上限
        2. 自适应采样：根据节点重要性在上限内动态调整实际采样数
        
        Args:
            node_idx: Index of the node
            neighbors: Array of neighbor indices
            hop_level: [NEW] 当前hop层级 (0-indexed)
            
        Returns:
            Sampled neighbor list
        """
        if len(neighbors) == 0:
            return []
        
 # ========== 第一步：分层采样 - 确定当前层的最大邻居数上限 ==========
        if self.use_hierarchical_sampling and hop_level < len(self.layer_sample_rates):
            layer_rate = self.layer_sample_rates[hop_level]
            # 基于层级采样率计算最大邻居数
            layer_max_neighbors = int(self.max_neighbors_per_node * layer_rate)
            layer_max_neighbors = max(5, layer_max_neighbors)  # 至少保留5个邻居
        else:
            layer_max_neighbors = self.max_neighbors_per_node
        
        if len(neighbors) <= layer_max_neighbors:
            return neighbors.tolist()
        
    # ========== 第二步：自适应采样 - 在上限内根据节点重要性动态调整 ==========
        if self.use_adaptive_sampling:
            importance = self.node_importance[node_idx]# 节点重要性 [0, 1]
            min_neighbors = max(5, layer_max_neighbors // 5)
            # 重要节点获得更多邻居，不重要节点获得较少邻居
            num_samples = int(min_neighbors + (layer_max_neighbors - min_neighbors) * importance)
            num_samples = min(num_samples, len(neighbors))
        else:
            num_samples = min(layer_max_neighbors, len(neighbors))
        
        import random
        return random.sample(neighbors.tolist(), num_samples)
    
    def sample(self, drug_indices: torch.Tensor, adr_indices: torch.Tensor) -> Tuple[torch.Tensor, Dict, Dict]:
        """Sample K-hop subgraph from batch drug and ADR nodes.
        
        [MODIFIED] 支持分层采样策略
        
        标准K-hop采样策略（限制邻居数量，防止节点爆炸）：
        1. 从batch节点开始，BFS收集K-hop邻居
        2. [NEW] 每层使用不同的采样率 (分层采样)
        3. 每个节点限制最大邻居数（max_neighbors_per_node）
        4. 限制总节点数（max_nodes）防止死机
        
        Args:
            drug_indices: Batch drug indices (local indices, 0 to n_drugs-1)
            adr_indices: Batch ADR indices (local indices, 0 to n_adrs-1)
            
        Returns:
            edge_index: Subgraph edge indices [2, num_edges]
            node_mapping: Dict mapping subgraph node idx -> global node idx
            metadata: Dict with subgraph information (包含node_types数组用于GPU加速)
        """
        # 优化：直接在CPU上处理，但使用numpy加速
        if drug_indices.is_cuda:
            drug_indices = drug_indices.cpu()
        if adr_indices.is_cuda:
            adr_indices = adr_indices.cpu()
        
        drug_indices = drug_indices.numpy()
        adr_indices = adr_indices.numpy()
        
        # Convert to global indices
        seed_drug_nodes = list(set(drug_indices.tolist()))
        seed_adr_nodes = list(set(self.graph_builder.adr_offset + idx for idx in adr_indices))
        
        # 确保 max_nodes 至少能容纳所有种子节点
        num_seed_nodes = len(seed_drug_nodes) + len(seed_adr_nodes)
        if self.max_nodes < num_seed_nodes:
            import warnings
            warnings.warn(f"max_nodes ({self.max_nodes}) is less than number of seed nodes ({num_seed_nodes}). "
                         f"Adjusting max_nodes to {num_seed_nodes + 50} to ensure all seeds are included.")
            effective_max_nodes = num_seed_nodes + 50
        else:
            effective_max_nodes = self.max_nodes
        
        # BFS to collect K-hop neighbors
        visited = set(seed_drug_nodes + seed_adr_nodes)
        current_level = set(seed_drug_nodes + seed_adr_nodes)
        all_nodes = set(seed_drug_nodes + seed_adr_nodes)
        
        # [NEW] 记录每个节点的hop层级，用于分层聚合
        node_hop_level = {node: 0 for node in all_nodes}
        
        # 预先计算节点类型映射
        node_type_cache = {}
        for node in seed_drug_nodes + seed_adr_nodes:
            node_type_cache[node] = self.graph_builder.get_node_type(node)
        
        for hop in range(self.k_hop):
            next_level = set()
            for node in current_level:
                neighbors = self.graph_builder.get_neighbors(node)
                # [MODIFIED] 传递hop层级给采样函数
                neighbors = self._adaptive_sample_neighbors(node, neighbors, hop_level=hop)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        all_nodes.add(neighbor)
                        # [NEW] 记录节点的hop层级
                        node_hop_level[neighbor] = hop + 1
                        if neighbor not in node_type_cache:
                            node_type_cache[neighbor] = self.graph_builder.get_node_type(neighbor)
            
            current_level = next_level
            if not current_level:
                break
            
            if len(all_nodes) >= effective_max_nodes:
                break
        
        # Limit subgraph size if needed
        if len(all_nodes) > effective_max_nodes:
            # Convert to numpy arrays for faster operations
            all_nodes_array = np.array(list(all_nodes))
            seed_nodes_array = np.array(seed_drug_nodes + seed_adr_nodes)
            
            # Priority-based node selection
            node_priorities = np.zeros(len(all_nodes_array))
            node_to_idx = {node: idx for idx, node in enumerate(all_nodes_array)}
            
            # Assign priorities: seeds=3, 1-hop neighbors=2, 2-hop neighbors=1
            for i, node in enumerate(all_nodes_array):
                if node in seed_nodes_array:
                    node_priorities[i] = 3
                elif node_hop_level.get(node, 0) == 1:
                    node_priorities[i] = 2
                else:
                    node_priorities[i] = 1
            
            # Add importance scores for tie-breaking
            if self.use_adaptive_sampling:
                importance_scores = np.array([self.node_importance[node] for node in all_nodes_array])
                node_priorities += importance_scores * 0.1  # Small weight for tie-breaking
            
            # Select top nodes by priority
            top_indices = np.argsort(node_priorities)[::-1][:effective_max_nodes]
            all_nodes = set(all_nodes_array[top_indices].tolist())
        # Create subgraph node mapping
        subgraph_nodes = sorted(list(all_nodes))
        node_mapping = {sub_idx: global_idx for sub_idx, global_idx in enumerate(subgraph_nodes)}
        reverse_mapping = {global_idx: sub_idx for sub_idx, global_idx in enumerate(subgraph_nodes)}
        
        # 构建子图的边列表
        subgraph_edges = []
        
        for global_src in subgraph_nodes:
            neighbors = self.graph_builder.get_neighbors(global_src)
            valid_neighbors = neighbors[np.isin(neighbors, subgraph_nodes)]
            
            if len(valid_neighbors) > 0:
                sub_src = reverse_mapping[global_src]
                for global_dst in valid_neighbors:
                    sub_dst = reverse_mapping[global_dst]
                    subgraph_edges.append((sub_src, sub_dst))
        
        # 转换为edge_index格式
        if subgraph_edges:
            edge_index = torch.tensor(list(zip(*subgraph_edges)), dtype=torch.long).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 预先计算所有节点类型
        node_types_list = [node_type_cache.get(n, self.graph_builder.get_node_type(n)) for n in subgraph_nodes]
        
        # [NEW] 计算各节点的hop层级 (用于分层聚合)
        node_hop_levels = [node_hop_level.get(n, 0) for n in subgraph_nodes]
        
        # 构建元数据字典
        metadata = {
            'num_nodes': len(subgraph_nodes),
            'num_edges': len(subgraph_edges),
            'seed_drug_nodes': [reverse_mapping.get(n, -1) for n in seed_drug_nodes],
            'seed_adr_nodes': [reverse_mapping.get(n, -1) for n in seed_adr_nodes],
            'node_types': node_types_list,
            'node_type_cache': node_type_cache,
            'node_hop_levels': node_hop_levels,  # [NEW] 节点hop层级信息
            'layer_sample_rates': self.layer_sample_rates  # [NEW] 采样率配置
        }
        
        return edge_index, node_mapping, metadata

