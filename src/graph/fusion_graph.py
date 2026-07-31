# fusion_graph.py
# run: pip install torch torch_geometric

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

logger = logging.getLogger(__name__)

# change this once we know the actual embedding dim from the other module
PLACEHOLDER_DIM = 128


def build_fusion_graph(
    positions: Tensor,
    track_ids: Tensor,
    embeddings: Optional[
        Tensor
    ] = None,  # optional, since we might not have real ones yet and need to use zeros
    radius: float = 1.0,
    max_num_neighbors: int = 32,
    loop: bool = False,
    batch: Optional[Tensor] = None,
    placeholder_dim: int = PLACEHOLDER_DIM,
) -> Data:
    # fill in zero embeddings if we don't have real ones yet
    embeddings = _get_embeddings(positions, embeddings, placeholder_dim)

    _check_inputs(positions, embeddings, track_ids, batch)

    N = positions.size(0)

    # edge case: empty input, just return an empty graph
    if N == 0:
        logger.warning("got empty input, returning empty graph")
        D = embeddings.size(1)
        return Data(
            x=torch.empty(0, 3 + D, dtype=embeddings.dtype, device=embeddings.device),
            edge_index=torch.empty(2, 0, dtype=torch.long, device=positions.device),
            pos=positions,
            track_ids=track_ids,
            batch=batch,
            has_real_embeddings=False,
        )

    # build edges based on spatial distance
    edge_index: Tensor = radius_graph(
        x=positions,
        r=radius,
        batch=batch,
        loop=loop,
        max_num_neighbors=max_num_neighbors,
    )

    logger.debug("graph: %d nodes, %d edges (r=%.2f)", N, edge_index.size(1), radius)

    # concat positions + embeddings into node features
    # need to cast to same dtype first or torch complains (fp16 vs fp32 stuff)
    dtype = torch.promote_types(positions.dtype, embeddings.dtype)
    node_feats: Tensor = torch.cat(
        [positions.to(dtype), embeddings.to(dtype)],
        dim=-1,
    )

    return Data(
        x=node_feats,
        edge_index=edge_index,
        pos=positions,
        track_ids=track_ids,
        batch=batch,
        has_real_embeddings=embeddings.any().item(),
    )


def compute_identity_embedding(
    track_embeddings: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    if track_embeddings.ndim != 2:
        raise ValueError(
            f"expected 2D tensor (T, D), got shape {tuple(track_embeddings.shape)}"
        )

    T, D = track_embeddings.shape

    # no frames = return zeros, don't crash
    if T == 0:
        logger.warning("empty track, returning zero embedding")
        return torch.zeros(
            D, dtype=track_embeddings.dtype, device=track_embeddings.device
        )

    if weights is not None:
        if weights.shape != (T,):
            raise ValueError(f"weights should be ({T},), got {tuple(weights.shape)}")
        if (weights < 0).any():
            raise ValueError("weights can't be negative")

        weight_sum = weights.sum()
        if weight_sum == 0:
            # all weights zero means just do plain mean
            logger.warning("all weights are zero, falling back to mean")
            return track_embeddings.mean(dim=0)

        # weighted average
        w = weights / weight_sum
        return (w.unsqueeze(-1) * track_embeddings).sum(dim=0)

    # plain mean if no weights given
    return track_embeddings.mean(dim=0)


def group_embeddings_by_track(
    track_ids: Tensor,
    embeddings: Optional[Tensor] = None,
    placeholder_dim: int = PLACEHOLDER_DIM,
) -> dict[int, Tensor]:
    N = track_ids.size(0)

    if N == 0:
        return {}

    # reuse the same placeholder logic as everywhere else
    embeddings = _get_embeddings(
        torch.empty(N, 3),  # dummy, just need shape for N
        embeddings,
        placeholder_dim,
    )

    unique_ids = track_ids.unique()

    # boolean mask slice per track - cleaner than looping
    return {int(tid): embeddings[track_ids == tid] for tid in unique_ids}


def build_fusion_graphs_batch(
    positions_list: list[Tensor],
    track_ids_list: list[Tensor],
    embeddings_list: Optional[list[Optional[Tensor]]] = None,
    radius: float = 1.0,
    max_num_neighbors: int = 32,
    loop: bool = False,
    placeholder_dim: int = PLACEHOLDER_DIM,
) -> Batch:
    n = len(positions_list)

    if len(track_ids_list) != n:
        raise ValueError("positions_list and track_ids_list must be same length")

    if embeddings_list is None:
        embeddings_list = [None] * n
    elif len(embeddings_list) != n:
        raise ValueError("embeddings_list must match length of positions_list")

    graphs = [
        build_fusion_graph(
            positions=pos,
            track_ids=tids,
            embeddings=emb,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            loop=loop,
            placeholder_dim=placeholder_dim,
        )
        for pos, tids, emb in zip(positions_list, track_ids_list, embeddings_list)
    ]

    return Batch.from_data_list(graphs)


# --- internal helpers, not part of the public API ---


def _get_embeddings(
    positions: Tensor,
    embeddings: Optional[Tensor],
    placeholder_dim: int,
) -> Tensor:
    """
    Returns embeddings if we have them, otherwise returns a zero tensor.
    This is the only place that touches the placeholder logic, so when
    real embeddings arrive we only update the caller - nothing here changes.
    """
    if embeddings is not None:
        return embeddings

    N = positions.size(0)
    logger.debug("no embeddings, using zeros (N=%d, D=%d)", N, placeholder_dim)
    return torch.zeros(
        N, placeholder_dim, dtype=positions.dtype, device=positions.device
    )


def _check_inputs(
    positions: Tensor,
    embeddings: Tensor,
    track_ids: Tensor,
    batch: Optional[Tensor],
) -> None:
    """basic shape checks so we catch problems early"""
    if positions.ndim != 2 or positions.size(1) != 3:
        raise ValueError(f"positions should be (N, 3), got {tuple(positions.shape)}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings should be (N, D), got {tuple(embeddings.shape)}")
    N = positions.size(0)
    if embeddings.size(0) != N:
        raise ValueError(
            f"positions and embeddings have different N: {N} vs {embeddings.size(0)}"
        )
    if track_ids.shape != (N,):
        raise ValueError(f"track_ids should be ({N},), got {tuple(track_ids.shape)}")
    if batch is not None and batch.shape != (N,):
        raise ValueError(f"batch should be ({N},), got {tuple(batch.shape)}")


# quick test - run this file directly to check everything works
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(42)

    N, D = 50, 128
    pos = torch.randn(N, 3)
    tids = torch.arange(N)

    # test without embeddings
    g1 = build_fusion_graph(pos, tids, embeddings=None, radius=1.5)
    print(f"\nno embeddings -> x={g1.x.shape}, real={g1.has_real_embeddings}")

    # test with real embeddings
    emb = torch.randn(N, D)
    g2 = build_fusion_graph(pos, tids, embeddings=emb, radius=1.5)
    print(f"real embeddings -> x={g2.x.shape}, real={g2.has_real_embeddings}")

    # identity embedding
    ident = compute_identity_embedding(torch.randn(10, D), weights=torch.rand(10))
    print(f"\nidentity embedding shape: {ident.shape}")

    # group by track
    groups = group_embeddings_by_track(tids[:10], embeddings=None)
    print(f"grouped tracks: {list(groups.keys())}")

    # batched - one scene has embeddings, one doesn't
    batched = build_fusion_graphs_batch(
        positions_list=[torch.randn(20, 3), torch.randn(30, 3)],
        track_ids_list=[torch.arange(20), torch.arange(30)],
        embeddings_list=[torch.randn(20, D), None],
        radius=1.5,
    )
    print(f"\nbatched: {batched.num_graphs} graphs, {batched.num_nodes} total nodes")

    # empty input shouldn't crash
    g_empty = build_fusion_graph(torch.empty(0, 3), torch.empty(0, dtype=torch.long))
    print(f"empty graph: {g_empty}")
