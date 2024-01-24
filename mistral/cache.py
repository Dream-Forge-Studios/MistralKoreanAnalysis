import torch
from typing import List, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 3  # (W, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)
        
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3 # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk = torch.split(xk, self.metadata.seqlens)
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        interleaved_k = interleave_list(cache_k, xk)
        interleaved_v = interleave_list(cache_v, xv)

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """

        if self.kv_seqlens is None:
            """
            seqlens = [5, 7, 3, 6]일 때 
            kv_seqlens = [0, 0, 0, 0]
            """
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        """
        mask를 적용할 곳 표시
        mask = [[False, True, True], [True, False, True]]
        """
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        #to_cache_mask = torch.tensor([False, True, True, True, False, True])
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        #[2, 2], True의 개수
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        """
        seqpos = [0, 5, 12, 15], seqlens = [5, 7, 3, 6]
        첫 번째 시퀀스: torch.arange(0, 0 + 5) → [0, 1, 2, 3, 4]
        두 번째 시퀀스: torch.arange(5, 5 + 7) → [5, 6, 7, 8, 9, 10, 11]
        세 번째 시퀀스: torch.arange(12, 12 + 3) → [12, 13, 14]
        네 번째 시퀀스: torch.arange(15, 15 + 6) → [15, 16, 17, 18, 19, 20]
        positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        """
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(device=self.device, dtype=torch.long)
        """
        seqlens = [5, 7, 3, 6]
        첫 번째 시퀀스: [0, 0, 0, 0, 0]
        두 번째 시퀀스: [1, 1, 1, 1, 1, 1, 1]
        세 번째 시퀀스: [2, 2, 2]
        네 번째 시퀀스: [3, 3, 3, 3, 3, 3]
        batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3])
        """
        batch_idx = torch.tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device, dtype=torch.long)
        """
        각 토큰이 슬라이딩 윈도우 기반의 캐시 시스템 내에서 어떤 위치에 저장되어야 하는지
        각 배치가 서로 겹치지 않도록 조정
        """
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        #시작 프롬프트인 경우
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            """
            BlockDiagonal: 마스크가 대각선을 중심으로 여러 개의 블록으로 구성
            Causal: 토큰이 자신보다 이전에 나타난 토큰들과만 상호작용할 수 있도록 제한
            """
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)]
            ).make_local_attention_from_bottomright(self.sliding_window)
        #모든 시퀀스가 하나의 토큰만 포함하는 경우
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
            )

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
