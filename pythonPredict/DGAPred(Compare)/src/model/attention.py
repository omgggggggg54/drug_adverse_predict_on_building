"""
Attention mechanisms for molecular graph neural networks
Simplified version adapted from chemprop attention modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class MultiBondFastAttention(nn.Module):
    """Bond level self-attention block (Fastformer) in message passing phase"""
    
    def __init__(self, hidden_size: int, num_heads: int = 6, dropout: float = 0.0):
        super(MultiBondFastAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5
        
        self.weight_alpha = nn.Parameter(torch.randn(self.att_size))
        self.weight_beta = nn.Parameter(torch.randn(self.att_size))
        self.weight_r = nn.Linear(self.att_size, self.att_size, bias=False)
        
        self.W_b_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()
    
    def forward(self, message: torch.Tensor, b_scope: list) -> torch.Tensor:
        """
        Args:
            message: Hidden states [batch_num_bonds, hidden_size]
            b_scope: List of tuples (start_bond_index, num_bonds) for each molecule
        Returns:
            Attended bond features [batch_num_bonds, hidden_size]
        """
        bond_vecs = []
        for i, (b_start, b_size) in enumerate(b_scope):
            if i == 0:
                bond_vecs.append(self.cached_zero_vector)
            
            cur_bond_message = message.narrow(0, b_start, b_size)  # [num_bonds, hidden]
            cur_bond_message_size = cur_bond_message.size()
            
            # Project to Q, K, V
            b_q = self.W_b_q(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            b_k = self.W_b_k(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            b_v = self.W_b_v(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            
            b_q = b_q.transpose(0, 1)  # [num_heads, num_bonds, att_size]
            b_k = b_k.transpose(0, 1)
            b_v = b_v.transpose(0, 1)
            h, n, d = b_q.shape
            
            # Calculate global query
            alpha_weight = torch.mul(b_q, self.weight_alpha) * self.scale_factor
            alpha_weight = F.softmax(alpha_weight, dim=-1)
            global_query = torch.mul(alpha_weight, b_q)
            global_query = torch.sum(global_query, dim=1)  # [num_heads, att_size]
            
            # Model interaction between global query and key
            repeat_global_query = einops.repeat(global_query, 'h d -> h copy d', copy=n)
            p = torch.mul(repeat_global_query, b_k)
            beta_weight = torch.mul(p, self.weight_beta) * self.scale_factor
            beta_weight = F.softmax(beta_weight, dim=-1)
            global_key = torch.mul(beta_weight, p)
            global_key = torch.sum(global_key, dim=1)
            
            # Key-value interaction
            key_value_interaction = torch.einsum('hd,hnd->hnd', global_key, b_v)
            key_value_interaction_out = self.weight_r(key_value_interaction)
            att_b_h = key_value_interaction_out + b_q
            
            att_b_h = self.act_func(att_b_h)
            att_b_h = self.dropout_layer(att_b_h)
            att_b_h = att_b_h.transpose(0, 1).contiguous()
            att_b_h = att_b_h.view(cur_bond_message_size[0], self.num_heads * self.att_size)
            att_b_h = self.W_b_o(att_b_h)
            
            att_b_h = att_b_h.unsqueeze(dim=0)
            att_b_h = self.norm(att_b_h)
            att_b_h = att_b_h.squeeze(dim=0)
            
            bond_vecs.extend(att_b_h)
        
        bond_vecs = torch.stack(bond_vecs, dim=0)
        return bond_vecs


class MultiBondAttention(nn.Module):
    """Bond level self-attention block (Transformer) in message passing phase"""
    
    def __init__(self, hidden_size: int, num_heads: int = 6, dropout: float = 0.0):
        super(MultiBondAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5
        
        self.W_b_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_b_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()
    
    def forward(self, message: torch.Tensor, b_scope: list) -> torch.Tensor:
        """
        Args:
            message: Hidden states [batch_num_bonds, hidden_size]
            b_scope: List of tuples (start_bond_index, num_bonds) for each molecule
        Returns:
            Attended bond features [batch_num_bonds, hidden_size]
        """
        bond_vecs = []
        for i, (b_start, b_size) in enumerate(b_scope):
            if i == 0:
                bond_vecs.append(self.cached_zero_vector)
            
            cur_bond_message = message.narrow(0, b_start, b_size)
            cur_bond_message_size = cur_bond_message.size()
            
            b_q = self.W_b_q(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            b_k = self.W_b_k(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            b_v = self.W_b_v(cur_bond_message).view(cur_bond_message_size[0], self.num_heads, self.att_size)
            
            b_q = b_q.transpose(0, 1)  # [num_heads, num_bonds, att_size]
            b_k = b_k.transpose(0, 1).transpose(1, 2)  # [num_heads, att_size, num_bonds]
            b_v = b_v.transpose(0, 1)
            
            att_b_w = torch.matmul(b_q, b_k)  # [num_heads, num_bonds, num_bonds]
            att_b_w = F.softmax(att_b_w * self.scale_factor, dim=2)
            att_b_h = torch.matmul(att_b_w, b_v)  # [num_heads, num_bonds, att_size]
            
            att_b_h = self.act_func(att_b_h)
            att_b_h = self.dropout_layer(att_b_h)
            att_b_h = att_b_h.transpose(0, 1).contiguous()
            att_b_h = att_b_h.view(cur_bond_message_size[0], self.num_heads * self.att_size)
            att_b_h = self.W_b_o(att_b_h)
            
            att_b_h = att_b_h.unsqueeze(dim=0)
            att_b_h = self.norm(att_b_h)
            att_b_h = att_b_h.squeeze(dim=0)
            
            bond_vecs.extend(att_b_h)
        
        bond_vecs = torch.stack(bond_vecs, dim=0)
        return bond_vecs


class MultiAtomAttention(nn.Module):
    """Atom level self-attention block (Transformer) in readout phase"""
    
    def __init__(self, hidden_size: int, num_heads: int = 6, dropout: float = 0.0,
                 adjacency: bool = False, distance: bool = False, coulomb: bool = False,
                 f_scale: float = 1.0, normalize_matrices: bool = False, device: str = 'cpu'):
        super(MultiAtomAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.att_size = self.hidden_size // self.num_heads
        self.scale_factor = self.att_size ** -0.5
        
        self.adjacency = adjacency
        self.distance = distance
        self.coulomb = coulomb
        self.f_scale = f_scale
        self.normalize_matrices = normalize_matrices
        self.device = torch.device(device) if isinstance(device, str) else device
        
        self.W_a_q = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_k = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_v = nn.Linear(self.hidden_size, self.num_heads * self.att_size, bias=False)
        self.W_a_o = nn.Linear(self.num_heads * self.att_size, self.hidden_size)
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=True)
        
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = nn.ReLU()
    
    def forward(self, cur_hiddens: torch.Tensor, i: int, 
                f_adj: list = None, f_dist: list = None, f_clb: list = None):
        """
        Args:
            cur_hiddens: Hidden states [num_atoms, hidden_size]
            i: Molecule index
            f_adj: Adjacency matrices list
            f_dist: Distance matrices list
            f_clb: Coulomb matrices list
        Returns:
            Attended atom features [num_atoms, hidden_size]
        """
        cur_hiddens_size = cur_hiddens.size()
        
        a_q = self.W_a_q(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_k = self.W_a_k(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        a_v = self.W_a_v(cur_hiddens).view(cur_hiddens_size[0], self.num_heads, self.att_size)
        
        a_q = a_q.transpose(0, 1)  # [num_heads, num_atoms, att_size]
        a_k = a_k.transpose(0, 1).transpose(1, 2)  # [num_heads, att_size, num_atoms]
        a_v = a_v.transpose(0, 1)
        
        att_a_w = torch.matmul(a_q, a_k)  # [num_heads, num_atoms, num_atoms]
        
        # Add matrix information if available
        if self.adjacency and f_adj is not None:
            mol_adj = torch.tensor(f_adj[i], dtype=torch.float32, device=self.device)
            att_a_w[0] = att_a_w[0] + self.f_scale * mol_adj
            att_a_w[1] = att_a_w[1] + self.f_scale * mol_adj
        
        if self.distance and f_dist is not None:
            mol_dist = torch.tensor(f_dist[i], dtype=torch.float32, device=self.device)
            if self.normalize_matrices:
                mol_dist = F.softmax(mol_dist, dim=1)
            att_a_w[2] = att_a_w[2] + self.f_scale * mol_dist
            att_a_w[3] = att_a_w[3] + self.f_scale * mol_dist
        
        if self.coulomb and f_clb is not None:
            mol_clb = torch.tensor(f_clb[i], dtype=torch.float32, device=self.device)
            if self.normalize_matrices:
                mol_clb = F.softmax(mol_clb, dim=1)
            att_a_w[4] = att_a_w[4] + self.f_scale * mol_clb
            att_a_w[5] = att_a_w[5] + self.f_scale * mol_clb
        
        att_a_w = F.softmax(att_a_w * self.scale_factor, dim=2)
        att_a_h = torch.matmul(att_a_w, a_v)  # [num_heads, num_atoms, att_size]
        
        att_a_h = self.act_func(att_a_h)
        att_a_h = self.dropout_layer(att_a_h)
        att_a_h = att_a_h.transpose(0, 1).contiguous()
        att_a_h = att_a_h.view(cur_hiddens_size[0], self.num_heads * self.att_size)
        att_a_h = self.W_a_o(att_a_h)
        
        att_a_h = att_a_h.unsqueeze(dim=0)
        att_a_h = self.norm(att_a_h)
        mol_vec = att_a_h.squeeze(dim=0)
        
        return mol_vec


class SublayerConnection(nn.Module):
    """Residual connection with dropout"""
    
    def __init__(self, dropout: float):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, original: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """Apply residual connection"""
        return original + self.dropout(attention)
