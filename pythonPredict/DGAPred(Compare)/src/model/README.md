# ChemProp Encoder with Attention Mechanisms

Enhanced ChemProp encoder with bond and atom attention mechanisms, identical to the reference implementation in `chemprop/models/mpn_att.py`.

## Features

- **Bond-level Attention**: Transformer or Fastformer attention in message passing
- **Atom-level Attention**: Attention in readout phase with optional matrix integration
- **Residual Connections**: For both bond and atom attention blocks
- **Multiple Aggregation Methods**: mean, sum, or norm
- **Matrix Support**: Adjacency, distance, and coulomb matrices

## Usage

### Basic Usage (No Attention)

```python
from src.model.chemprop_encoder import ChemPropDrugEncoder

encoder = ChemPropDrugEncoder(
    output_dim=256,
    hidden_size=300,
    depth=3,
    dropout=0.0
)

drug_embeddings = encoder(
    smiles_list=['CCO', 'c1ccccc1'],
    adj_list=None,
    dist_list=None,
    clb_list=None
)
```

### With Bond Attention (Transformer)

```python
encoder = ChemPropDrugEncoder(
    output_dim=256,
    hidden_size=300,
    depth=3,
    dropout=0.0,
    bond_attention=True,      # Enable bond attention
    num_heads=6               # Number of attention heads
)
```

### With Bond Fast Attention (Fastformer)

```python
encoder = ChemPropDrugEncoder(
    output_dim=256,
    hidden_size=300,
    depth=3,
    dropout=0.0,
    bond_fast_attention=True,  # Enable fast attention
    num_heads=6
)
```

### With Atom Attention + Matrices

```python
encoder = ChemPropDrugEncoder(
    output_dim=256,
    hidden_size=300,
    depth=3,
    dropout=0.0,
    atom_attention=True,      # Enable atom attention
    num_heads=6,
    adjacency=True,           # Use adjacency matrix
    distance=True,            # Use distance matrix
    coulomb=True,             # Use coulomb matrix
    f_scale=1.0,              # Scale factor for matrices
    normalize_matrices=False, # Whether to normalize matrices
    device='cuda'             # Computing device
)

drug_embeddings = encoder(
    smiles_list=smiles_list,
    adj_list=adj_matrices,    # Required if adjacency=True
    dist_list=dist_matrices,  # Required if distance=True
    clb_list=clb_matrices     # Required if coulomb=True
)
```

### Full Configuration (All Features)

```python
encoder = ChemPropDrugEncoder(
    output_dim=256,
    hidden_size=300,
    depth=3,
    dropout=0.1,
    bond_attention=True,      # or bond_fast_attention=True
    atom_attention=True,
    num_heads=6,
    aggregation='mean',       # 'mean', 'sum', or 'norm'
    aggregation_norm=100,     # Used when aggregation='norm'
    adjacency=True,
    distance=True,
    coulomb=True,
    f_scale=1.0,
    normalize_matrices=False,
    device='cuda'
)
```

## Parameters

### ChemPropDrugEncoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dim` | int | required | Output embedding dimension |
| `hidden_size` | int | 300 | Hidden layer dimension |
| `depth` | int | 3 | Message passing depth |
| `dropout` | float | 0.0 | Dropout ratio |
| `bond_attention` | bool | False | Use bond-level Transformer attention |
| `bond_fast_attention` | bool | False | Use bond-level Fastformer attention |
| `atom_attention` | bool | False | Use atom-level attention |
| `num_heads` | int | 6 | Number of attention heads |
| `aggregation` | str | 'mean' | Aggregation method ('mean', 'sum', 'norm') |
| `aggregation_norm` | int | 100 | Normalization constant for 'norm' aggregation |
| `adjacency` | bool | False | Use adjacency matrix in atom attention |
| `distance` | bool | False | Use distance matrix in atom attention |
| `coulomb` | bool | False | Use coulomb matrix in atom attention |
| `f_scale` | float | 1.0 | Scale factor for matrices |
| `normalize_matrices` | bool | False | Normalize matrices with softmax |
| `device` | str | 'cpu' | Computing device |

## Architecture

```
Input: SMILES strings
  ↓
Molecular Graph Construction (BatchMolGraph)
  ↓
Message Passing (bond-centered)
  ├─ Optional: Bond Attention
  └─ Optional: Bond Residual Connection
  ↓
Atom-level Aggregation
  ↓
Readout
  ├─ Optional: Atom Attention (with matrices)
  ├─ Optional: Atom Residual Connection
  └─ Aggregation (mean/sum/norm)
  ↓
Output: mol_vecs [batch_size, hidden_size]
  ↓
Projection
  ↓
Output: drug_embeddings [batch_size, output_dim]
```

## Notes

- Only `mol_vecs` is returned (no `tom_vecs` or `batch_indices`)
- Bond attention and bond fast attention are mutually exclusive
- When using atom attention with matrices, ensure matrices are provided
- The implementation is identical to `chemprop/models/mpn_att.py` but simplified for inference
