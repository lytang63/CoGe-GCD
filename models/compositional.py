"""CoGe-GCD compositional token refinement.

This file contains the token assignment, evidence passing and geometric
calibration kernel used by the project. The numerical operations and default
parameters are kept identical to the reference implementation; only the
public terminology is expressed in paper language.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patch_neighbors(height, width, device):
    token_count = height * width
    indices = torch.arange(token_count, device=device)
    row, col = indices // width, indices % width
    up = torch.where(row > 0, indices - width, indices)
    down = torch.where(row < height - 1, indices + width, indices)
    left = torch.where(col > 0, indices - 1, indices)
    right = torch.where(col < width - 1, indices + 1, indices)
    neighbors = torch.stack([up, down, left, right], dim=1)
    valid = (neighbors != indices.unsqueeze(1)).float()
    degree = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    return neighbors, valid, degree


def _column_normalize(assignments, eps=1e-12):
    return assignments / (assignments.sum(dim=0, keepdim=True) + eps)


def calibrate_memberships(scores, tokens, height, width, temperature=1.0, eps=1e-12):
    """Apply the paper's proximity, similarity and continuity calibration."""
    device = scores.device
    batch, token_count, primitive_count = scores.shape
    memberships = scores

    neighbors, valid, degree = _patch_neighbors(height, width, device)
    neighbors = neighbors.unsqueeze(0).expand(batch, -1, -1)
    valid = valid.unsqueeze(0).expand(batch, -1, -1)
    degree = degree.unsqueeze(0).expand(batch, -1, -1)

    neighbor_memberships = torch.gather(
        memberships.unsqueeze(2).expand(-1, -1, 4, -1),
        1,
        neighbors.unsqueeze(-1).expand(-1, -1, -1, primitive_count),
    )
    proximity_memberships = (neighbor_memberships * valid.unsqueeze(-1)).sum(2) / degree

    normalized_tokens = F.normalize(tokens, dim=-1)
    neighbor_tokens = torch.gather(
        normalized_tokens.unsqueeze(2).expand(-1, -1, 4, -1),
        1,
        neighbors.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1]),
    )
    cosine_similarity = (normalized_tokens.unsqueeze(2) * neighbor_tokens).sum(-1)
    relation_weights = torch.clamp(cosine_similarity, min=0.0) * valid
    relation_weights = relation_weights / (relation_weights.sum(2, keepdim=True) + eps)
    similarity_memberships = (neighbor_memberships * relation_weights.unsqueeze(-1)).sum(2)

    proximity_neighbors = torch.gather(
        proximity_memberships.unsqueeze(2).expand(-1, -1, 4, -1),
        1,
        neighbors.unsqueeze(-1).expand(-1, -1, -1, primitive_count),
    )
    continuity_memberships = (proximity_neighbors * relation_weights.unsqueeze(-1)).sum(2)

    denominator = (memberships * memberships).sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    score_proximity = (memberships * proximity_memberships).sum(dim=(1, 2), keepdim=True) / denominator
    score_similarity = (memberships * similarity_memberships).sum(dim=(1, 2), keepdim=True) / denominator
    score_continuity = (memberships * continuity_memberships).sum(dim=(1, 2), keepdim=True) / denominator

    weight_proximity = torch.clamp(1.0 - score_proximity, min=0.0)
    weight_similarity = torch.clamp(1.0 - score_similarity, min=0.0)
    weight_continuity = torch.clamp(1.0 - score_continuity, min=0.0)
    normalizer = weight_proximity + weight_similarity + weight_continuity + eps
    weight_proximity = weight_proximity / normalizer
    weight_similarity = weight_similarity / normalizer
    weight_continuity = weight_continuity / normalizer

    calibrated = memberships + 0.1 * (
        weight_proximity * proximity_memberships
        + weight_similarity * similarity_memberships
        + weight_continuity * continuity_memberships
    )
    calibration_weights = {
        "w_p": weight_proximity.view(batch).clone(),
        "w_s": weight_similarity.view(batch).clone(),
        "w_c": weight_continuity.view(batch).clone(),
    }
    return calibrated, calibration_weights


class PrimitiveAssignment(nn.Module):
    """Generate image-conditioned primitive memberships for patch tokens."""

    def __init__(self, node_dim, num_primitives, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        self.num_heads = num_heads
        self.num_primitives = num_primitives
        self.head_dim = node_dim // num_heads
        self.context = context
        self.prototype_base = nn.Parameter(torch.Tensor(num_primitives, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, num_primitives * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, num_primitives * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'. Expected one of: 'mean', 'max', 'both'.")
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, tokens):
        batch, token_count, dim = tokens.shape
        if self.context == "mean":
            context = tokens.mean(dim=1)
        elif self.context == "max":
            context, _ = tokens.max(dim=1)
        else:
            context = torch.cat([tokens.mean(dim=1), tokens.max(dim=1)[0]], dim=-1)
        primitive_offsets = self.context_net(context).view(batch, self.num_primitives, dim)
        primitives = self.prototype_base.unsqueeze(0) + primitive_offsets

        projected_tokens = self.pre_head_proj(tokens)
        token_heads = projected_tokens.view(batch, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        primitive_heads = primitives.view(batch, self.num_primitives, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        token_heads_flat = token_heads.reshape(batch * self.num_heads, token_count, self.head_dim)
        primitive_heads_flat = primitive_heads.reshape(batch * self.num_heads, self.num_primitives, self.head_dim).transpose(1, 2)
        logits = torch.bmm(token_heads_flat, primitive_heads_flat) / self.scaling
        logits = logits.view(batch, self.num_heads, token_count, self.num_primitives).mean(dim=1)
        logits = self.dropout(logits)
        calibrated_logits, weights = calibrate_memberships(logits, tokens, int(math.sqrt(token_count)), int(math.sqrt(token_count)))
        return F.softmax(calibrated_logits, dim=1), weights


class EvidenceConsolidation(nn.Module):
    """Pass evidence token -> primitive -> token with a residual update."""

    def __init__(self, embed_dim, num_primitives=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()
        # Attribute names are retained for checkpoint compatibility.
        self.edge_generator = PrimitiveAssignment(embed_dim, num_primitives, num_heads, dropout, context)
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

    def forward(self, tokens):
        memberships, weights = self.edge_generator(tokens)
        primitive_evidence = torch.bmm(memberships.transpose(1, 2), tokens)
        primitive_evidence = self.edge_proj(primitive_evidence)
        refined_tokens = torch.bmm(memberships, primitive_evidence)
        refined_tokens = self.node_proj(refined_tokens)
        return refined_tokens + tokens, weights, memberships


class CompositionalPerception(nn.Module):
    """Paper-facing entry point for the unchanged CoGe-GCD token module."""

    def __init__(self, embed_dim, num_primitives=16, num_heads=8, dropout=0.1, context="both"):
        super().__init__()
        self.embed_dim = embed_dim
        self.last_calibration_weights = {}
        # Keep the historical attribute name so existing checkpoints load.
        self.hgnn = EvidenceConsolidation(embed_dim, num_primitives, num_heads, dropout, context)

    def forward(self, tokens):
        refined_tokens, weights, memberships = self.hgnn(tokens)
        self.last_calibration_weights = weights
        self.memberships = memberships
        return refined_tokens

