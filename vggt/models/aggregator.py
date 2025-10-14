# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
                
    @staticmethod
    def _build_global_attn_mask_from_adj(adj: torch.Tensor, S: int, P: int, device: torch.device, dtype: torch.dtype):
        """
        adj: (S,S) bool/0-1 with diagonal True (ALLOW)
        returns (1,1,N,N) additive mask for SDPA where N=S*P
        """
        adj = adj.bool()
        adj = adj.clone()
        adj.fill_diagonal_(True)

        N = S * P
        frame_ids = torch.arange(S, device=device).repeat_interleave(P)  # (N,)
        allow = adj[frame_ids][:, frame_ids]  # (N,N) bool

        mask = torch.zeros((N, N), device=device, dtype=dtype)
        mask[~allow] = float("-inf")
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
    
    def forward(self, images: torch.Tensor, attn_mask: torch.Tensor | None = None) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        
        
                # ---- 1) Helpers ----
        def _connected_components_undirected(adj_bool_2d: torch.Tensor):
            """
            adj_bool_2d: (S, S) bool, undirected connectivity (True=edge)
            returns list[list[int]] of node indices
            """
            S_ = adj_bool_2d.shape[0]
            seen = torch.zeros(S_, dtype=torch.bool, device=adj_bool_2d.device)
            comps = []
            for s in range(S_):
                if seen[s]:
                    continue
                # DFS/BFS
                stack = [s]
                seen[s] = True
                comp = [s]
                while stack:
                    u = stack.pop()
                    # neighbors with True edges
                    nbrs = torch.where(adj_bool_2d[u])[0]
                    for v in nbrs.tolist():
                        if not seen[v]:
                            seen[v] = True
                            stack.append(v)
                            comp.append(v)
                comps.append(comp)
            return comps

        def compute_tight_components(adj_bool_2d: torch.Tensor,
                                    mutual: bool = True,
                                    window: int | None = None,
                                    jaccard_tau: float | None = None,
                                    k_core_k: int | None = None):
            """
            adj_bool_2d: (S,S) bool, True=edge (directed or undirected)
            Returns: list[list[int]] connected components after pruning.
            """

            A = adj_bool_2d.clone()

            S = A.shape[0]
            # 1) symmetrize
            A = A | A.T                # weakly undirected (either direction)
            if mutual:
                A = A & A.T            # require i<->j (mutual)

            # 2) force self-edges
            i = torch.arange(S, device=A.device)
            A[i, i] = True

            # 3) optional: local window
            if window is not None:
                # keep |i-j| <= window
                idx = torch.arange(S, device=A.device)
                band = (idx[:, None] - idx[None, :]).abs() <= window
                A = A & band

            # 4) optional: Jaccard prune (neighborhood similarity)
            if jaccard_tau is not None:
                # avoid diag in sim (but keep diag in A)
                N = A.clone()
                N[i, i] = False
                inter = (N[:, None, :] & N[None, :, :]).sum(dim=-1)                  # (S,S)
                union = (N[:, None, :] | N[None, :, :]).sum(dim=-1).clamp_min_(1)
                jacc = inter.float() / union.float()
                keep = jacc >= jaccard_tau
                A = (A & keep) | torch.diag(torch.ones(S, dtype=torch.bool, device=A.device))

            # 5) optional: k-core prune
            if k_core_k is not None:
                deg = A.sum(dim=1) - 1  # exclude self
                active = torch.ones(S, dtype=torch.bool, device=A.device)
                changed = True
                while changed:
                    drop = active & (deg < k_core_k)
                    changed = bool(drop.any().item())
                    active = active & ~drop
                    if changed:
                        # zero edges for dropped nodes
                        A[drop, :] = False
                        A[:, drop] = False
                        A[i, i] = True
                        deg = A.sum(dim=1) - 1

            # Connected components on the pruned, undirected graph
            S_ = S
            seen = torch.zeros(S_, dtype=torch.bool, device=A.device)
            comps = []
            for s in range(S_):
                if seen[s]:
                    continue
                stack = [int(s)]
                seen[s] = True
                comp = [int(s)]
                while stack:
                    u = stack.pop()
                    nbrs = torch.where(A[u])[0]
                    for v in nbrs.tolist():
                        if not seen[v]:
                            seen[v] = True
                            stack.append(int(v))
                            comp.append(int(v))
                comps.append(comp)
            return comps
        
        def merge_singletons_to_next(groups, S):
            """
            groups: list[list[int]]  (components over nodes 0..S-1)
            S: total number of frames/nodes

            Rule: if a component has size 1 (singleton {i}), merge i into the group
            that contains (i+1) % S. If *all* components are singletons, merge them all
            into a single group [0,1,...,S-1].
            """
            # keep only non-singletons first
            new_groups = [set(g) for g in groups if len(g) > 1]
            if not new_groups:
                # all singletons -> one big group
                return [list(range(S))]

            # map node -> group index for the current (non-singleton) groups
            node_to_gid = {}
            for gid, g in enumerate(new_groups):
                for v in g:
                    node_to_gid[v] = gid

            # collect singletons in sorted order for determinism
            singles = sorted([g[0] for g in groups if len(g) == 1])

            for i in singles:
                # find the next index (wrap) that already sits in a non-singleton group
                t = (i + 1) % S
                hopped = 0
                while t not in node_to_gid and hopped < S:
                    t = (t + 1) % S
                    hopped += 1
                if hopped >= S:
                    # should not happen because we handled "all singletons" above,
                    # but just in case, attach to the first group
                    target_gid = 0
                else:
                    target_gid = node_to_gid[t]

                # merge i into that group
                new_groups[target_gid].add(i)
                node_to_gid[i] = target_gid

            # return as sorted lists (optional: keep original ordering of groups)
            return [sorted(list(g)) for g in new_groups]


        def _frame_indices_for_groups(groups, P_, device):
            """
            groups: list of list[int] (frame ids)
            returns list[1D LongTensor] (token indices within each group)
            """
            idx_list = []
            for g in groups:
                if len(g) == 1:
                    s = g[0]
                    idx = torch.arange(s * P_, (s + 1) * P_, device=device)
                else:
                    idx = torch.cat(
                        [torch.arange(s * P_, (s + 1) * P_, device=device) for s in g],
                        dim=0
                    )
                idx_list.append(idx)
            return idx_list
        def _normalize_allow_mask(attn_mask, B, S, P, device):
            """
            Normalize various attention mask shapes/semantics into allow_bool: (B, N, N), True = allowed.

            Supports:
            - (S,S)              segment-level, bool or numeric
            - (B,S,S)            per-batch segment-level
            - (N,N)              token-level
            - (B,N,N)            per-batch token-level
            - (B,1,N,N)          SDPA-style head dim squeezed later

            Semantics:
            - bool, token-level (N,N) or (B,N,N) or (B,1,N,N):
                assumed SDPA-style (True = DISALLOW) → inverted to allow
            - bool, segment-level (S,S) or (B,S,S):
                assumed adjacency-style (True = ALLOW) → used as-is
            - numeric:
                assumed 0 = ALLOW, nonzero = disallow
            """
            N = S * P
            m = attn_mask

            if m is None:
                return None  # caller will handle the "one big group" case

            # ----- Handle segment-level first -----
            # (S,S) -> (B,S,S) -> lift to (B,N,N)
            if m.dim() == 2 and m.shape == (S, S):
                # bool: interpret as adjacency (True = ALLOW)
                if m.dtype == torch.bool:
                    seg_allow = m
                else:
                    # numeric: 0 = allow
                    seg_allow = (m == 0)
                seg_allow = seg_allow.unsqueeze(0).expand(B, S, S)  # (B,S,S)

                # Lift to token-level by repeating each segment entry P×P
                # (B,S,S) -> (B,S,1,S,1) -> expand to (B,S,P,S,P) -> reshape (B,N,N)
                seg_allow = seg_allow.unsqueeze(2).unsqueeze(4)               # (B,S,1,S,1)
                seg_allow = seg_allow.expand(-1, -1, P, -1, P)                # (B,S,P,S,P)
                allow_bool = seg_allow.reshape(B, N, N)                        # (B,N,N)
                return allow_bool.to(device)

            # (B,S,S) -> lift to (B,N,N)
            if m.dim() == 3 and m.shape[1:] == (S, S):
                if m.dtype == torch.bool:
                    seg_allow = m  # adjacency True = ALLOW
                else:
                    seg_allow = (m == 0)
                seg_allow = seg_allow.unsqueeze(2).unsqueeze(4)               # (B,S,1,S,1)
                seg_allow = seg_allow.expand(-1, -1, P, -1, P)                # (B,S,P,S,P)
                allow_bool = seg_allow.reshape(B, N, N)
                return allow_bool.to(device)

            # ----- Token-level shapes -----
            # (N,N) -> (B,N,N)
            if m.dim() == 2 and m.shape == (N, N):
                if m.dtype == torch.bool:
                    # SDPA-style (True = DISALLOW) at token-level → invert
                    allow_bool = ~m
                else:
                    allow_bool = (m == 0)
                allow_bool = allow_bool.unsqueeze(0).expand(B, N, N)
                return allow_bool.to(device)

            # (B,N,N)
            if m.dim() == 3 and m.shape[1:] == (N, N):
                if m.dtype == torch.bool:
                    allow_bool = ~m
                else:
                    allow_bool = (m == 0)
                return allow_bool.to(device)

            # (B,1,N,N) -> squeeze head dim
            if m.dim() == 4 and m.shape[-2:] == (N, N):
                m = m.squeeze(1)  # (B,N,N)
                if m.dtype == torch.bool:
                    allow_bool = ~m
                else:
                    allow_bool = (m == 0)
                return allow_bool.to(device)

            raise ValueError(f"Unsupported attn_mask shape {tuple(m.shape)} for S={S}, P={P}")
        
        def _build_affinity(W, tau=0.5, topk=5, window=None):
            """
            W: (S,S) float in [0,1] (or any real scores). Returns symmetric weighted affinity A (S,S).
            - keeps edges with W>=tau
            - adds per-row top-k neighbors as a safety net
            - optionally restricts to a temporal window
            """
            S = W.shape[0]
            device = W.device

            # base thresholded edges (bool)
            A = (W >= tau)

            # optional temporal window
            if window is not None:
                i = torch.arange(S, device=device)
                band = (i[:, None] - i[None, :]).abs() <= window
                A = A & band

            # k-NN safety net (work in float, exclude self, respect window if given)
            k = min(max(1, topk), max(0, S - 1))
            if k > 0:
                W_knn = W.clone()
                W_knn.fill_diagonal_(-float("inf"))
                if window is not None:
                    W_knn = W_knn.masked_fill(~band, -float("inf"))
                idx = torch.topk(W_knn, k=k, dim=1).indices           # (S,k)
                knn = torch.zeros((S, S), dtype=torch.bool, device=device)
                knn.scatter_(1, idx, True)
                A = A | knn                                           # out-of-place OR

            # symmetrize OUT-OF-PLACE to avoid overlap issue
            A = A | A.t()

            # ensure diagonal
            A.fill_diagonal_(True)

            # weight with original W on the allowed support
            return W * A.to(W.dtype)


        def _laplacian_eigs(A, kmax=10):
            S = A.shape[0]
            d = A.sum(dim=1).clamp_min_(1e-8)
            Dm12 = torch.diag_embed(d.rsqrt())
            Asym = Dm12 @ A @ Dm12
            Lsym = torch.eye(S, device=A.device, dtype=A.dtype) - Asym
            evals, evecs = torch.linalg.eigh(Lsym.to(torch.float32))      # ascending
            kmax = min(kmax, S)
            return evals[:kmax], evecs[:, :kmax]

        def _choose_k_by_eigengap(evals, kmin=2, kmax=None):
            S = evals.numel()
            if kmax is None: kmax = min(10, S)
            kmax = min(kmax, S-1)
            if kmax < 1: return 1
            gaps = (evals[1:kmax+1] - evals[0:kmax])    # Δλ
            k = int(torch.argmax(gaps).item()) + 1
            return max(kmin, min(k, kmax))

        def _kmeans_torch(X, k, iters=30, seed=0):
            N = X.shape[0]
            g = torch.Generator(device=X.device).manual_seed(seed)
            C = X[torch.randperm(N, generator=g, device=X.device)[:k]].clone()
            for _ in range(iters):
                x2 = (X*X).sum(dim=1, keepdim=True)
                c2 = (C*C).sum(dim=1).unsqueeze(0)
                d2 = x2 + c2 - 2.0*(X @ C.t())
                labels = torch.argmin(d2, dim=1)
                newC = torch.zeros_like(C)
                counts = torch.bincount(labels, minlength=k).clamp_min_(1).float().unsqueeze(1)
                newC.index_add_(0, labels, X)
                newC /= counts
                if torch.allclose(newC, C, atol=1e-5): break
                C = newC
            return labels

        def _labels_to_groups(labels):
            groups = {}
            for i, g in enumerate(labels.tolist()):
                groups.setdefault(g, []).append(i)
            return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]

        def merge_singletons_to_next(groups, S):
            non_single = [set(g) for g in groups if len(g) > 1]
            if not non_single:
                return [list(range(S))]
            node_to_gid = {v: gid for gid, g in enumerate(non_single) for v in g}
            singles = sorted([g[0] for g in groups if len(g) == 1])
            for i in singles:
                t = (i + 1) % S
                hops = 0
                while t not in node_to_gid and hops < S:
                    t = (t + 1) % S; hops += 1
                gid = node_to_gid.get(t, 0)
                non_single[gid].add(i); node_to_gid[i] = gid
            return [sorted(list(g)) for g in non_single]

        def spectral_groups_from_W(W, tau=0.5, topk=3, window=None, k=None, kmin=2, kmax=6, ensure_no_singletons=True):
            S = W.shape[0]
            A = _build_affinity(W, tau=tau, topk=topk, window=window)
            evals, U = _laplacian_eigs(A, kmax=max(kmax, (k or 2)))
            if k is None:
                k = _choose_k_by_eigengap(evals, kmin=kmin, kmax=kmax)
            U_k = torch.nn.functional.normalize(U[:, :k], p=2, dim=1)
            labels = _kmeans_torch(U_k, k=k)
            groups = _labels_to_groups(labels)
            if ensure_no_singletons:
                groups = merge_singletons_to_next(groups, S)
            return groups


        def _pick_topM_bridges(groups, S, W, M=2):
            """
            For each group g, pick up to M external frames with highest average W to g.
            Returns: list[list[int]] of bridge frame ids per group.
            """
            out = []
            for g in groups:
                gset = set(g)
                cand = [j for j in range(S) if j not in gset]
                if not cand or M <= 0:
                    out.append([])
                    continue
                with torch.no_grad():
                    gi = torch.tensor(sorted(g), device=W.device, dtype=torch.long)
                    cj = torch.tensor(cand, device=W.device, dtype=torch.long)
                    # avg similarity to the group for each candidate
                    scores = W[cj][:, gi].mean(dim=1)    # (|cand|,)
                    topk = min(M, cj.numel())
                    sel = torch.topk(scores, k=topk, dim=0).indices
                    out.append(cj[sel].tolist())
            return out

        def _frame_indices_with_bridge_specials_rw(groups, bridges, S, P, patch_start_idx, device):
            """
            For each group:
            - write_idx: full tokens for group's frames (specials+patches)
            - read_idx:  write_idx + ONLY specials of each bridge frame
            Returns: list[(read_idx, write_idx)]
            """
            rw_list = []
            for g, br in zip(groups, bridges):
                # own frames (full)
                if len(g) == 1:
                    s = g[0]
                    write_idx = torch.arange(s*P, (s+1)*P, device=device)
                else:
                    write_idx = torch.cat([torch.arange(s*P, (s+1)*P, device=device) for s in g], dim=0)

                # bridge specials
                bridge_specs = []
                for f in br:
                    base = f * P
                    bridge_specs.append(torch.arange(base, base + patch_start_idx, device=device))
                bridge_specs = (torch.cat(bridge_specs) if bridge_specs else
                                torch.empty(0, dtype=torch.long, device=device))

                read_idx = torch.cat([write_idx, bridge_specs], dim=0)
                # (optional) dedupe if you worry about overlaps:
                # read_idx = torch.unique_consecutive(read_idx)
                rw_list.append((read_idx, write_idx))
            return rw_list


        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
            
        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape
        
    
        frame_idx = 0
        global_idx = 0
        output_list = []
  
        import time 
        start = time.perf_counter()
  

        # # ---- 2) Build per-batch groups from the (optional) attention mask ----
        per_batch_idx_lists = []
        per_batch_idx_lists_global = []

        groups_b = [list(range(S))]
        idx_list = _frame_indices_for_groups(groups_b, P, tokens.device)
        
        # rw_list_global = [(idx, idx) for idx in idx_list]
        # per_batch_idx_lists_global = [rw_list_global for _ in range(B)]
        
        per_batch_idx_lists_global = [idx_list for _ in range(B)]
            
        if attn_mask is None:
            # one big group containing all frames
            per_batch_idx_lists = per_batch_idx_lists_global
               
        else:
            # Normalize mask to (B, N, N) with True = "allowed"
            allow_bool = _normalize_allow_mask(attn_mask, B, S, P, tokens.device)  # (B,N,N), True=ALLOW

            # Vectorized reduction to segment graph:
            # reshape to (B, S, P, S, P) then "any" over the P×P blocks
            A = allow_bool.view(B, S, P, S, P)
            adj_ss = A.any(dim=(2, 4))                          # (B, S, S), True if ANY token pair allowed

            # Convert to undirected (weak connectivity): edge exists if i→j OR j→i
            adj_ss = adj_ss | adj_ss.transpose(-1, -2)

            # Ensure self-edges
            d = torch.arange(S, device=adj_ss.device)
            adj_ss[:, d, d] = True
            # print(adj_ss)
            # Build connected components per batch and expand to token indices
            for b in range(B):
                # groups = _connected_components_undirected(adj_ss[b])
                groups = compute_tight_components(
                    adj_ss[b],
                    mutual=True,        # require i<->j
                    window=None,           # keep only neighbors within ±2 frames (tune or None)
                    jaccard_tau=False,    # drop weak overlaps (0.1–0.4 typical), or None
                    k_core_k=False       # e.g., 2 or 3 to peel fringes, or None
                )
                
            
                # # groups = [[0,1,2], [3,4,5]]
                # groups = merge_singletons_to_next(groups, S)
                
                # tau = 0.5
                # groups = spectral_groups_from_W(
                #     attn_mask,
                #     tau=tau,          # base edge threshold
                #     topk=3,           # ensure each frame has at least a few neighbors
                #     window=None,      # or an integer if you want temporal banding
                #     k=None,           # let eigengap pick k
                #     kmin=2, kmax=6,
                #     ensure_no_singletons=True,
                # )

                # groups = [[0] + [x for x in group if x != 0] for group in groups]
                
                print(groups)
                
                idx_list = _frame_indices_for_groups(groups, P, tokens.device)
                # print(idx_list)
                per_batch_idx_lists.append(idx_list)
                
                
                # M = 2  # tune: 1–3 is usually enough
                # bridges = _pick_topM_bridges(groups, S, W=attn_mask, M=M)

                # rw_list = _frame_indices_with_bridge_specials_rw(
                #     groups, bridges, S, P, self.patch_start_idx, tokens.device
                # )
                # per_batch_idx_lists.append(rw_list)


       
        end = time.perf_counter()
        length = end - start

        print("Preprocessing Cross-Attention took", length, "seconds!")
        for _ in range(self.aa_block_num):
            tokens = tokens.contiguous()
            if pos is not None: pos = pos.contiguous()
            for attn_type in self.aa_order:
                
                if attn_type == "frame":
                    start = time.perf_counter()
                    
                    # print(tokens.shape, pos.shape)
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                    end = time.perf_counter()
                    length = end - start

                    # print("Frame Att iter took", length, "seconds!")
                elif attn_type == "global":
                    if (global_idx % 2) == 0:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, per_batch_idx_lists_global, B, S, P, C, global_idx, pos=pos
                        )
                    else:
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, per_batch_idx_lists, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
                
           

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
                
        
            
        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

  
    def _process_global_attention(self, tokens,per_batch_idx_lists, B, S, P, C, global_idx, pos=None, ):
        """
        tokens: (B, S*P, C) or (B, S, P, C)
        pos:    (B, S*P, 2)  or (B, S, P, 2) or None
        attn_mask:
        - None                          -> one group containing all S frames
        - (N,N) or (B,N,N) or (B,1,N,N) -> supports bool (True=disallow) OR numeric (0=allow)
        """

        # ---- 0) Normalize tokens/pos to (B, N, C) / (B, N, 2) ----
        N = S * P
        if tokens.shape != (B, N, C):
            tokens = tokens.view(B, S, P, C).reshape(B, N, C)
        if pos is not None and pos.shape != (B, N, 2):
            pos = pos.view(B, S, P, 2).reshape(B, N, 2)

        def run_grouped_blk(blk, tokens, pos, per_batch_idx_lists):
            """
            tokens: (B, N, C), pos: (B, N, 2) or None
            per_batch_idx_lists: list over b of list[LongTensor idx] (token indices per group)
            Returns updated tokens.
            """
            B, N, C = tokens.shape
            out = tokens.clone()

            # 1) Build a catalog of (b, idx, L)
            entries = []
            for b, idxs in enumerate(per_batch_idx_lists):
                for idx in idxs:
                    entries.append((b, idx, idx.numel()))
            # 2) Bucket by length L to batch calls
            from collections import defaultdict
            buckets = defaultdict(list)
            for b, idx, L in entries:
                buckets[L].append((b, idx))

            # 3) For each bucket length L, stack groups and run blk once
            for L, items in buckets.items():
                # gather groups
                x_list, p_list, locs = [], [], []
                for (b, idx) in items:
                    x_list.append(tokens[b:b+1].index_select(1, idx))            # (1, L, C)
                    if pos is not None:
                        p_list.append(pos[b:b+1].index_select(1, idx))           # (1, L, 2)
                    locs.append((b, idx))
                x_batch = torch.cat(x_list, dim=0)                                # (G, L, C)
                p_batch = torch.cat(p_list, dim=0) if pos is not None else None   # (G, L, 2) or None

                # single blk call for all groups of this length
                x_batch = blk(x_batch, pos=p_batch, attn_mask=None)               # (G, L, C)

                # scatter back
                for g, (b, idx) in enumerate(locs):
                    out[b:b+1].index_copy_(1, idx, x_batch[g:g+1])

            return out

        # def run_grouped_blk(blk, tokens, pos, per_batch_rw_lists):
        #     """
        #     tokens: (B, N, C), pos: (B, N, 2) or None
        #     per_batch_rw_lists: list over b of list[(read_idx, write_idx)]
        #     """
        #     B, N, C = tokens.shape
        #     out = tokens.clone()

        #     # catalog entries: (b, read_idx, write_idx, Lr, Lw)
        #     entries = []
        #     for b, pairs in enumerate(per_batch_rw_lists):
        #         for (read_idx, write_idx) in pairs:
        #             entries.append((b, read_idx, write_idx, read_idx.numel(), write_idx.numel()))

        #     # bucket by read length (so we can batch)
        #     from collections import defaultdict
        #     buckets = defaultdict(list)
        #     for b, r_idx, w_idx, Lr, Lw in entries:
        #         buckets[Lr].append((b, r_idx, w_idx, Lw))

        #     for Lr, items in buckets.items():
        #         # gather
        #         x_list, p_list, locs = [], [], []
        #         for (b, r_idx, w_idx, Lw) in items:
        #             x_list.append(tokens[b:b+1].index_select(1, r_idx))      # (1, Lr, C)
        #             if pos is not None:
        #                 p_list.append(pos[b:b+1].index_select(1, r_idx))     # (1, Lr, 2)
        #             locs.append((b, r_idx, w_idx, Lw))
        #         x_batch = torch.cat(x_list, dim=0)                            # (G, Lr, C)
        #         p_batch = torch.cat(p_list, dim=0) if pos is not None else None

        #         # run attention on full read set (group tokens + bridge specials)
        #         x_batch = blk(x_batch, pos=p_batch, attn_mask=None)           # (G, Lr, C)

        #         # scatter back ONLY the first Lw tokens (the group's own tokens)
        #         for g, (b, r_idx, w_idx, Lw) in enumerate(locs):
        #             out[b:b+1].index_copy_(1, w_idx, x_batch[g:g+1, :Lw, :])

        #     return out

        # ---- 3) Run grouped attention blocks (no mask inside each group) ----
        intermediates = []
        
            
        start = time.perf_counter()
        for _ in range(self.aa_block_size):
            blk = self.global_blocks[global_idx]
            tokens = run_grouped_blk(blk, tokens, pos, per_batch_idx_lists)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        end = time.perf_counter()
        length = end - start

        # print("Running cross attention took", length, "seconds!")
        
        return tokens, global_idx, intermediates

def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
