"""Graph construction and degree normalization module for LightGCN.

Builds the undirected bipartite user-item graph and computes the symmetrically
normalized adjacency operator:

    \\tilde{A} = D^{-1/2} A D^{-1/2}

where:
- A is the (M + N) x (M + N) bipartite adjacency matrix.
- D is the diagonal degree matrix with D_ii = sum_j A_ij.
- M is the number of users, N is the number of items.
- Nodes 0 .. M-1 represent users.
- Nodes M .. M+N-1 represent items.
"""

from typing import Tuple
import numpy as np
import scipy.sparse as sp
import torch


def compute_node_degrees(
    num_users: int,
    num_items: int,
    train_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute degrees for users, items, and total graph nodes.

    Args:
        num_users: Total number of users M.
        num_items: Total number of items N.
        train_pairs: Array of shape (num_edges, 2) containing [user_id, item_id].

    Returns:
        user_degrees: Array of shape (num_users,) containing user degrees |N_u|.
        item_degrees: Array of shape (num_items,) containing item degrees |N_i|.
        total_degrees: Array of shape (num_users + num_items,) containing all node degrees.
    """
    user_degrees = np.zeros(num_users, dtype=np.float32)
    item_degrees = np.zeros(num_items, dtype=np.float32)

    if len(train_pairs) > 0:
        # Deduplicate pairs to prevent multi-edge double counting
        unique_pairs = np.unique(train_pairs, axis=0)
        u_nodes = unique_pairs[:, 0]
        i_nodes = unique_pairs[:, 1]

        u_unique, u_counts = np.unique(u_nodes, return_counts=True)
        user_degrees[u_unique] = u_counts.astype(np.float32)

        i_unique, i_counts = np.unique(i_nodes, return_counts=True)
        item_degrees[i_unique] = i_counts.astype(np.float32)

    total_degrees = np.concatenate([user_degrees, item_degrees])
    return user_degrees, item_degrees, total_degrees


def build_sparse_adjacency(
    num_users: int,
    num_items: int,
    train_pairs: np.ndarray,
) -> sp.coo_matrix:
    """Construct the unnormalized bipartite adjacency matrix A in SciPy COO format.

    The adjacency matrix has structure:
        A = [[ 0,   R  ],
             [ R^T, 0  ]]
    where R is the (num_users x num_items) interaction matrix.

    Nodes:
        Users: indices [0, num_users - 1]
        Items: indices [num_users, num_users + num_items - 1]
    """
    total_nodes = num_users + num_items
    if len(train_pairs) == 0:
        return sp.coo_matrix((total_nodes, total_nodes), dtype=np.float32)

    unique_pairs = np.unique(train_pairs, axis=0)
    user_ids = unique_pairs[:, 0]
    item_ids = unique_pairs[:, 1]

    # Map item indices into the unified node index space: item_node = num_users + item_id
    item_nodes = item_ids + num_users

    # Bipartite undirected graph: (user -> item) and (item -> user)
    row = np.concatenate([user_ids, item_nodes])
    col = np.concatenate([item_nodes, user_ids])
    data = np.ones(len(row), dtype=np.float32)

    adj = sp.coo_matrix((data, (row, col)), shape=(total_nodes, total_nodes), dtype=np.float32)
    return adj


def build_normalized_adjacency(
    num_users: int,
    num_items: int,
    train_pairs: np.ndarray,
) -> sp.coo_matrix:
    """Compute the symmetrically normalized adjacency matrix \\tilde{A} = D^{-1/2} A D^{-1/2}.

    For an edge between user u and item i:
        \\tilde{A}_{u, M+i} = 1 / (sqrt(|N_u|) * sqrt(|N_i|))
        \\tilde{A}_{M+i, u} = 1 / (sqrt(|N_i|) * sqrt(|N_u|))

    Isolated nodes with degree 0 are safely assigned 0 (D_ii^{-1/2} = 0) to avoid
    division by zero or NaN propagation.
    """
    adj = build_sparse_adjacency(num_users, num_items, train_pairs)
    total_nodes = num_users + num_items

    if adj.nnz == 0:
        return sp.coo_matrix((total_nodes, total_nodes), dtype=np.float32)

    # Compute row degrees
    degree = np.array(adj.sum(axis=1)).flatten()

    # D^{-1/2} with safe handling of isolated nodes
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.power(degree, -0.5, dtype=np.float32)
    d_inv_sqrt[np.isinf(d_inv_sqrt) | np.isnan(d_inv_sqrt)] = 0.0

    # Symmetric degree scaling: D^{-1/2} A D^{-1/2}
    # For each non-zero entry (r, c), value becomes d_inv_sqrt[r] * A_{r,c} * d_inv_sqrt[c]
    rows = adj.row
    cols = adj.col
    values = d_inv_sqrt[rows] * adj.data * d_inv_sqrt[cols]

    norm_adj = sp.coo_matrix((values, (rows, cols)), shape=(total_nodes, total_nodes), dtype=np.float32)
    return norm_adj


def scipy_coo_to_torch_sparse(coo: sp.coo_matrix) -> torch.Tensor:
    """Convert a SciPy COO sparse matrix to a coalesced PyTorch sparse COO tensor."""
    if coo.nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).coalesce()

    # Ensure COO format
    coo = coo.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.from_numpy(indices).to(torch.long)
    v = torch.from_numpy(coo.data).to(torch.float32)
    shape = torch.Size(coo.shape)

    sparse_tensor = torch.sparse_coo_tensor(i, v, shape, dtype=torch.float32).coalesce()
    return sparse_tensor


def build_torch_normalized_adjacency(
    num_users: int,
    num_items: int,
    train_pairs: np.ndarray,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """High-level utility to build and return \\tilde{A} directly as a PyTorch sparse tensor."""
    sp_norm_adj = build_normalized_adjacency(num_users, num_items, train_pairs)
    torch_adj = scipy_coo_to_torch_sparse(sp_norm_adj).to(device)
    return torch_adj
