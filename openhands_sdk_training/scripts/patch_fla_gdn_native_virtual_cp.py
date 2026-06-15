#!/usr/bin/env python3
"""Patch FLA/Megatron for native GDN virtual-chunk context parallelism.

Megatron/MCA uses load-balanced head-tail CP: rank r owns virtual chunks
``[r, 2 * CP - r - 1]``. FLA's operator-level CP assumes each rank owns a
single contiguous sequence interval. The conservative ADP bridge redistributes
full q/k/v/gate tensors into that contiguous layout before calling FLA.

This patch adds an opt-in native path controlled by
``ADP_MCA_GDN_CP_NATIVE_VCHUNK=1``. It keeps q/k/v tensors in Megatron's
head-tail order and teaches FLA's compact CP exchanges to operate on the two
local virtual chunks:

* causal conv exchanges only W-1 predecessor tokens per virtual chunk;
* gated-delta forward/backward exchanges only compact recurrent summaries;
* local FLA kernels see two local variable-length sequences via
  ``cu_seqlens=[0, chunk_len, 2 * chunk_len]``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


MEGATRON_IMPORT_MARKER = "# ADP patch: import FLACPContext for native virtual CP."
MEGATRON_CONTEXT_MARKER = "# ADP patch: build native virtual-chunk GDN CP context."
MEGATRON_SKIP_INPUT_EXCHANGE_MARKER = (
    "# ADP patch: native virtual CP keeps Megatron head-tail input layout."
)
MEGATRON_SKIP_OUTPUT_EXCHANGE_MARKER = (
    "# ADP patch: native virtual CP keeps Megatron head-tail output layout."
)
CONV_HELPERS_MARKER = "# ADP patch: native virtual-chunk conv CP helpers."
CONV_PREP_MARKER = "# ADP patch: prepare native virtual-chunk conv initial states."
CONV_CTX_MARKER = "# ADP patch: save native virtual-chunk conv metadata."
CONV_SIGNATURE_MARKER = "# ADP patch: accept native virtual-chunk conv correction metadata."
CONV_CORRECT_MARKER = "# ADP patch: correct native virtual-chunk conv dx."
CONV_BWD_CALL_MARKER = (
    "# ADP patch: pass native virtual-chunk conv metadata to backward correction."
)
GDN_HELPERS_MARKER = "# ADP patch: native virtual-chunk GDN CP helpers."
GDN_FWD_MARKER = "# ADP patch: native virtual-chunk GDN forward state exchange."
GDN_BWD_MARKER = "# ADP patch: native virtual-chunk GDN backward state exchange."
GDN_COMPRESS_MARKER = "# ADP patch: preserve native virtual-chunk initial states."
GDN_EXPAND_MARKER = "# ADP patch: preserve native virtual-chunk expanded states."
GDN_FWD_MULTISEQ_MARKER = "# ADP patch: fuse native virtual GDN forward summaries."
GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_MARKER = (
    "# ADP patch: add multi-sequence native virtual GDN backward summary kernel support."
)
GDN_BWD_KERNEL_MULTISEQ_OFFSET_MARKER = (
    "# ADP patch: write native virtual GDN backward summaries per sequence."
)
GDN_BWD_MULTISEQ_MARKER = "# ADP patch: fuse native virtual GDN backward summaries."
GDN_BWD_ORIGINAL_CALL_MARKER = "# ADP patch: pass single-sequence flag to original GDN backward summary."


def patch_module(
    module: str,
    old: str,
    new: str,
    marker: str,
    description: str,
    *,
    missing_ok: bool = False,
    old_missing_ok: bool = False,
) -> int:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError) as exc:
        if missing_ok:
            print(f"Skipping optional {description}: {module} import failed ({exc}).")
            return 0
        raise

    if spec is None or spec.origin is None:
        if missing_ok:
            print(f"Skipping optional {description}: could not find {module}.")
            return 0
        print(f"Could not find {module} on PYTHONPATH", file=sys.stderr)
        return 1

    path = pathlib.Path(spec.origin)
    text = path.read_text()
    if marker in text:
        print(f"Already patched {description}: {path}")
        return 0
    if old not in text:
        if old_missing_ok:
            print(f"Skipping {description}: expected block not found in {path}")
            return 0
        print(f"Expected {description} block not found in {path}", file=sys.stderr)
        return 1

    path.write_text(text.replace(old, new, 1))
    print(f"Patched {description}: {path}")
    return 0


MEGATRON_IMPORT_OLD = """    from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
"""

MEGATRON_IMPORT_NEW = """    from fla.modules.conv.causal_conv1d import causal_conv1d as fla_causal_conv1d
    from fla.modules.l2norm import l2norm
    from fla.ops.cp import FLACPContext, build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # ADP patch: import FLACPContext for native virtual CP.
"""

MEGATRON_CONTEXT_OLD = """        cp_context = None
        if self.cp_size > 1:
            if batch != 1:
                raise NotImplementedError(
                    "ADP experimental GDN context parallelism currently expects microbatch size 1."
                )
            global_seq_len = seq_len * self.cp_size
            cu_seqlens_cpu = torch.tensor([0, global_seq_len], dtype=torch.long)
            cu_seqlens = cu_seqlens_cpu.to(device=hidden_states.device, non_blocking=True)
            # ADP patch: pass FLA context-parallel metadata through GDN.
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens,
                group=self.cp_group,
                conv1d_kernel_size=self.conv_kernel_dim,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
"""

MEGATRON_CONTEXT_NEW = """        cp_context = None
        if self.cp_size > 1:
            if batch != 1:
                raise NotImplementedError(
                    "ADP experimental GDN context parallelism currently expects microbatch size 1."
                )
            use_native_virtual_cp = _adp_env_flag("ADP_MCA_GDN_CP_NATIVE_VCHUNK")
            if use_native_virtual_cp:
                if seq_len % 2 != 0:
                    raise RuntimeError(
                        "Native virtual-chunk GDN CP expects Megatron head-tail local "
                        f"sequence length to split into two equal chunks, got {seq_len}."
                    )
                chunk_len = seq_len // 2
                cu_seqlens_cpu = torch.tensor([0, chunk_len, seq_len], dtype=torch.long)
                cu_seqlens = cu_seqlens_cpu.to(device=hidden_states.device, non_blocking=True)
                cp_rank = dist.get_rank(self.cp_group)
                cp_context = FLACPContext(
                    group=self.cp_group,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                    is_last_rank=False,
                    pre_num_ranks=None,
                    is_first_rank=False,
                    post_num_ranks=None,
                    conv1d_kernel_size=self.conv_kernel_dim,
                    pre_num_conv_tokens=self.conv_kernel_dim - 1,
                )
                cp_context.adp_virtual_cp = True
                cp_context.adp_virtual_chunks = (cp_rank, 2 * self.cp_size - cp_rank - 1)
                cp_context.adp_virtual_chunk_count = 2 * self.cp_size
                cp_context.adp_virtual_chunk_len = chunk_len
                # ADP patch: build native virtual-chunk GDN CP context.
            else:
                global_seq_len = seq_len * self.cp_size
                cu_seqlens_cpu = torch.tensor([0, global_seq_len], dtype=torch.long)
                cu_seqlens = cu_seqlens_cpu.to(device=hidden_states.device, non_blocking=True)
                # ADP patch: pass FLA context-parallel metadata through GDN.
                cp_context = build_cp_context(
                    cu_seqlens=cu_seqlens,
                    group=self.cp_group,
                    conv1d_kernel_size=self.conv_kernel_dim,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
"""

MEGATRON_UNDO_OLD = """        original_cp_positions = None
        sorted_cp_positions = None
        use_full_gather_cp_bridge = _adp_env_flag("ADP_MCA_GDN_CP_FULL_GATHER_BRIDGE")
        if cp_context is not None:
            if qkvzba.shape[1] != seq_len:
                raise RuntimeError(
                    "Unexpected GDN sequence shape after input projection: "
                    f"got {qkvzba.shape[1]}, expected {seq_len}."
                )
            if use_full_gather_cp_bridge:
                original_cp_positions = _adp_cp_local_positions(seq_len, self.cp_group, qkvzba.device)
                qkvzba, sorted_cp_positions = _adp_cp_undo_load_balancing(
                    qkvzba, original_cp_positions, self.cp_group, dim=1
                )
            else:
                qkvzba = _adp_cp_head_tail_to_contiguous(qkvzba, self.cp_group, dim=1)
                # ADP patch: exchange head-tail CP chunks into contiguous GDN layout.
            # ADP patch: convert load-balanced CP layout to contiguous GDN chunks.
"""

MEGATRON_UNDO_NEW = """        original_cp_positions = None
        sorted_cp_positions = None
        use_full_gather_cp_bridge = _adp_env_flag("ADP_MCA_GDN_CP_FULL_GATHER_BRIDGE")
        use_native_virtual_cp = bool(getattr(cp_context, "adp_virtual_cp", False))
        if cp_context is not None:
            if qkvzba.shape[1] != seq_len:
                raise RuntimeError(
                    "Unexpected GDN sequence shape after input projection: "
                    f"got {qkvzba.shape[1]}, expected {seq_len}."
                )
            if use_native_virtual_cp:
                # ADP patch: native virtual CP keeps Megatron head-tail input layout.
                pass
            elif use_full_gather_cp_bridge:
                original_cp_positions = _adp_cp_local_positions(seq_len, self.cp_group, qkvzba.device)
                qkvzba, sorted_cp_positions = _adp_cp_undo_load_balancing(
                    qkvzba, original_cp_positions, self.cp_group, dim=1
                )
            else:
                qkvzba = _adp_cp_head_tail_to_contiguous(qkvzba, self.cp_group, dim=1)
                # ADP patch: exchange head-tail CP chunks into contiguous GDN layout.
            # ADP patch: convert load-balanced CP layout to contiguous GDN chunks.
"""

MEGATRON_REDO_OLD = """        norm_out = norm_out.reshape(batch, seq_len, -1)
        if cp_context is not None:
            if use_full_gather_cp_bridge:
                norm_out = _adp_cp_redo_load_balancing(
                    norm_out, original_cp_positions, sorted_cp_positions, self.cp_group, dim=1
                )
            else:
                norm_out = _adp_cp_contiguous_to_head_tail(norm_out, self.cp_group, dim=1)
                # ADP patch: exchange contiguous GDN chunks back to head-tail CP layout.
            # ADP patch: restore load-balanced CP layout after GDN.
        norm_out = norm_out.transpose(0, 1).contiguous()
"""

MEGATRON_REDO_NEW = """        norm_out = norm_out.reshape(batch, seq_len, -1)
        if cp_context is not None:
            if use_native_virtual_cp:
                # ADP patch: native virtual CP keeps Megatron head-tail output layout.
                pass
            elif use_full_gather_cp_bridge:
                norm_out = _adp_cp_redo_load_balancing(
                    norm_out, original_cp_positions, sorted_cp_positions, self.cp_group, dim=1
                )
            else:
                norm_out = _adp_cp_contiguous_to_head_tail(norm_out, self.cp_group, dim=1)
                # ADP patch: exchange contiguous GDN chunks back to head-tail CP layout.
            # ADP patch: restore load-balanced CP layout after GDN.
        norm_out = norm_out.transpose(0, 1).contiguous()
"""

CONV_HELPERS_OLD = """from fla.ops.utils import prepare_chunk_indices


class CausalConv1dFunctionCP(torch.autograd.Function):
"""

CONV_HELPERS_NEW = """from fla.ops.utils import prepare_chunk_indices


def _adp_is_virtual_cp(context: FLACPContext | None) -> bool:
    return bool(getattr(context, "adp_virtual_cp", False))


def _adp_virtual_owner_rank(virtual_chunk: int, cp_size: int) -> int:
    return virtual_chunk if virtual_chunk < cp_size else 2 * cp_size - virtual_chunk - 1


def _adp_virtual_local_index(virtual_chunk: int, cp_size: int) -> int:
    owner = _adp_virtual_owner_rank(virtual_chunk, cp_size)
    return 0 if virtual_chunk == owner else 1


def _adp_prepare_virtual_initial_state_for_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    context: FLACPContext,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    W = weight.shape[-1]
    D = weight.shape[0]
    virtual_chunks = tuple(int(c) for c in context.adp_virtual_chunks)
    n_chunks = len(virtual_chunks)
    if x.dim() != 3 or x.shape[0] != 1:
        raise RuntimeError(f"Native virtual CP conv requires [1, T, D], got {x.shape}.")
    if x.shape[1] % n_chunks != 0:
        raise RuntimeError(
            "Native virtual CP conv expects equal local virtual chunks, "
            f"got local length {x.shape[1]} and {n_chunks} chunks."
        )

    cp_size = dist.get_world_size(group)
    chunk_len = x.shape[1] // n_chunks
    tail_len = W - 1
    x_2d = x.squeeze(0)
    tails = []
    for local_idx in range(n_chunks):
        start = local_idx * chunk_len
        tails.append(x_2d.narrow(0, start + chunk_len - tail_len, tail_len))
    tails = torch.stack(tails, dim=0).contiguous()
    gathered = torch.empty(cp_size, n_chunks, tail_len, D, device=x.device, dtype=x.dtype)
    dist.all_gather_into_tensor(gathered, tails, group=group)

    initial_state = torch.zeros(n_chunks, D, W, device=x.device, dtype=x.dtype)
    for local_idx, virtual_chunk in enumerate(virtual_chunks):
        if virtual_chunk == 0:
            continue
        prev_chunk = virtual_chunk - 1
        prev_owner = _adp_virtual_owner_rank(prev_chunk, cp_size)
        prev_local_idx = _adp_virtual_local_index(prev_chunk, cp_size)
        initial_state[local_idx, :, -tail_len:] = gathered[prev_owner, prev_local_idx].T
    return initial_state


def _adp_correct_virtual_dx_for_cp(
    dx: torch.Tensor,
    dh0: torch.Tensor | None,
    W: int,
    group: dist.ProcessGroup,
    virtual_chunks: tuple[int, ...],
) -> None:
    cp_size = dist.get_world_size(group)
    n_chunks = len(virtual_chunks)
    if dx.shape[1] % n_chunks != 0:
        raise RuntimeError(
            "Native virtual CP conv backward expects equal local virtual chunks, "
            f"got local length {dx.shape[1]} and {n_chunks} chunks."
        )
    D = dx.shape[-1]
    chunk_len = dx.shape[1] // n_chunks
    tail_len = W - 1
    total_virtual_chunks = 2 * cp_size

    d_initial = torch.zeros(n_chunks, tail_len, D, device=dx.device, dtype=dx.dtype)
    if dh0 is not None:
        for local_idx, virtual_chunk in enumerate(virtual_chunks):
            if virtual_chunk == 0:
                continue
            d_initial[local_idx] = dh0[local_idx, :, -tail_len:].T

    gathered = torch.empty(cp_size, n_chunks, tail_len, D, device=dx.device, dtype=dx.dtype)
    dist.all_gather_into_tensor(gathered, d_initial.contiguous(), group=group)

    for local_idx, virtual_chunk in enumerate(virtual_chunks):
        next_chunk = virtual_chunk + 1
        if next_chunk >= total_virtual_chunks:
            continue
        next_owner = _adp_virtual_owner_rank(next_chunk, cp_size)
        next_local_idx = _adp_virtual_local_index(next_chunk, cp_size)
        start = local_idx * chunk_len + chunk_len - tail_len
        dx[0, start : start + tail_len, :].add_(gathered[next_owner, next_local_idx])


# ADP patch: native virtual-chunk conv CP helpers.


class CausalConv1dFunctionCP(torch.autograd.Function):
"""

CONV_PREP_OLD = """        if group is None:
            return None

        W = weight.shape[-1]  # weight: [D, W]
"""

CONV_PREP_NEW = """        if group is None:
            return None
        if _adp_is_virtual_cp(context):
            # ADP patch: prepare native virtual-chunk conv initial states.
            return _adp_prepare_virtual_initial_state_for_cp(x, weight, context, group)

        W = weight.shape[-1]  # weight: [D, W]
"""

CONV_SIGNATURE_OLD = """        pre_num_conv_tokens: int = 0,
    ) -> None:
"""

CONV_SIGNATURE_NEW = """        pre_num_conv_tokens: int = 0,
        virtual_chunks: tuple[int, ...] | None = None,
        # ADP patch: accept native virtual-chunk conv correction metadata.
    ) -> None:
"""

CONV_CORRECT_OLD = """        if group is None:
            return

        D = dx.shape[-1]
"""

CONV_CORRECT_NEW = """        if group is None:
            return
        if virtual_chunks is not None:
            # ADP patch: correct native virtual-chunk conv dx.
            _adp_correct_virtual_dx_for_cp(dx, dh0, W, group, virtual_chunks)
            return

        D = dx.shape[-1]
"""

CONV_CTX_OLD = """        ctx.is_first_rank = cp_context.is_first_rank
        ctx.pre_num_conv_tokens = cp_context.pre_num_conv_tokens

        # Call original forward
"""

CONV_CTX_NEW = """        ctx.is_first_rank = cp_context.is_first_rank
        ctx.pre_num_conv_tokens = cp_context.pre_num_conv_tokens
        ctx.adp_virtual_chunks = (
            tuple(int(c) for c in cp_context.adp_virtual_chunks)
            if _adp_is_virtual_cp(cp_context)
            else None
        )
        # ADP patch: save native virtual-chunk conv metadata.

        # Call original forward
"""

CONV_BWD_OLD = """            is_first_rank=ctx.is_first_rank,
            pre_num_conv_tokens=ctx.pre_num_conv_tokens,
        )
"""

CONV_BWD_NEW = """            is_first_rank=ctx.is_first_rank,
            pre_num_conv_tokens=ctx.pre_num_conv_tokens,
            virtual_chunks=ctx.adp_virtual_chunks,
            # ADP patch: pass native virtual-chunk conv metadata to backward correction.
        )
"""

GDN_HELPERS_OLD = """def chunk_gated_delta_rule_fwd_h_pre_process(
"""

GDN_HELPERS_NEW = """def _adp_is_virtual_cp(context: FLACPContext | None) -> bool:
    return bool(getattr(context, "adp_virtual_cp", False))


def _adp_virtual_owner_rank(virtual_chunk: int, cp_size: int) -> int:
    return virtual_chunk if virtual_chunk < cp_size else 2 * cp_size - virtual_chunk - 1


def _adp_virtual_local_index(virtual_chunk: int, cp_size: int) -> int:
    owner = _adp_virtual_owner_rank(virtual_chunk, cp_size)
    return 0 if virtual_chunk == owner else 1


def _adp_virtual_summaries_in_global_order(
    gathered: torch.Tensor,
    context: FLACPContext,
) -> torch.Tensor:
    cp_size = dist.get_world_size(context.group)
    n_chunks = len(context.adp_virtual_chunks)
    if n_chunks != 2:
        raise RuntimeError(f"Expected two local virtual chunks, got {n_chunks}.")
    ordered = gathered.new_empty(2 * cp_size, *gathered.shape[2:])
    for rank in range(cp_size):
        ordered[rank].copy_(gathered[rank, 0])
        ordered[2 * cp_size - rank - 1].copy_(gathered[rank, 1])
    return ordered.contiguous()


def _adp_narrow_seq(tensor: torch.Tensor | None, start: int, length: int) -> torch.Tensor | None:
    return None if tensor is None else tensor.narrow(1, start, length)


def _adp_chunk_gated_delta_rule_fwd_h_pre_process_virtual(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext = None,
    transpose_state_layout: bool = False,
) -> torch.Tensor:
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    if transpose_state_layout:
        raise NotImplementedError("ADP native virtual CP has only been wired for K,V state layout.")

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    virtual_chunks = tuple(int(c) for c in context.adp_virtual_chunks)
    n_chunks = len(virtual_chunks)
    if B != 1 or T % n_chunks != 0:
        raise RuntimeError(
            "ADP native virtual GDN CP expects batch=1 and equal local chunks, "
            f"got B={B}, T={T}, n_chunks={n_chunks}."
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    chunk_len = T // n_chunks
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    summaries = k.new_zeros(n_chunks, HV, K, V + K, dtype=torch.float32)
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, n_chunks)
    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u if v is None else v,
        w=w,
        g=g,
        gk=gk,
        bg=bg,
        u=u,
        hm=summaries,
        cu_seqlens=None,
        T=chunk_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK1=BK,
        USE_EXP2=use_exp2,
        BLOCK_SIZE=BLOCK_SIZE,
        MULTI_SEQS=True,
    )

    gathered, _ = all_gather_into_tensor(summaries, group=context.group)
    virtual_summaries = _adp_virtual_summaries_in_global_order(gathered, context)
    initial_state = k.new_zeros(n_chunks, HV, K, V, dtype=torch.float32)
    for local_idx, virtual_chunk in enumerate(virtual_chunks):
        if virtual_chunk == 0:
            continue
        def grid_merge(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid_merge](
            h=initial_state[local_idx],
            ag_hm=virtual_summaries,
            pre_or_post_num_ranks=virtual_chunk,
            rank=virtual_chunk,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=False,
        )
    return initial_state


def _adp_chunk_gated_delta_rule_bwd_dhu_pre_process_virtual(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    use_exp2: bool = False,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, None]:
    assert dht is None, "When enable CP, the provided dht must be None."
    if transpose_state_layout:
        raise NotImplementedError("ADP native virtual CP has only been wired for K,V state layout.")

    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    virtual_chunks = tuple(int(c) for c in context.adp_virtual_chunks)
    n_chunks = len(virtual_chunks)
    if B != 1 or T % n_chunks != 0:
        raise RuntimeError(
            "ADP native virtual GDN CP expects batch=1 and equal local chunks, "
            f"got B={B}, T={T}, n_chunks={n_chunks}."
        )
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    chunk_len = T // n_chunks
    BT = 64
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    summaries = q.new_zeros(n_chunks, HV, K, V + K, dtype=torch.float32)
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, n_chunks)
    pre_process_bwd_kernel_merged[grid](
        q=q,
        k=k if bg is None else bg,
        w=w,
        g=g,
        gk=gk,
        do=do,
        dhm=summaries,
        dv=dv,
        cu_seqlens=None,
        scale=scale,
        T=chunk_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK1=BK,
        USE_EXP2=use_exp2,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BG=bg is not None,
        MULTI_SEQS=True,
    )

    gathered, _ = all_gather_into_tensor(summaries, group=context.group)
    virtual_summaries = _adp_virtual_summaries_in_global_order(gathered, context)
    dht = q.new_zeros(n_chunks, HV, K, V, dtype=torch.float32)
    total_virtual_chunks = int(context.adp_virtual_chunk_count)
    for local_idx, virtual_chunk in enumerate(virtual_chunks):
        post_num_chunks = total_virtual_chunks - virtual_chunk - 1
        if post_num_chunks <= 0:
            continue
        def grid_merge(meta): return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel[grid_merge](
            h=dht[local_idx],
            ag_hm=virtual_summaries,
            pre_or_post_num_ranks=post_num_chunks,
            rank=virtual_chunk,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=False,
        )
    return dht, None


# ADP patch: native virtual-chunk GDN CP helpers.


def chunk_gated_delta_rule_fwd_h_pre_process(
"""

GDN_FWD_OLD = """    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    rank = dist.get_rank(group=context.group)
"""

GDN_FWD_NEW = """    if context is None or context.group is None:
        return initial_state
    if _adp_is_virtual_cp(context):
        # ADP patch: native virtual-chunk GDN forward state exchange.
        return _adp_chunk_gated_delta_rule_fwd_h_pre_process_virtual(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            bg=bg,
            v=v,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            initial_state=initial_state,
            context=context,
            transpose_state_layout=transpose_state_layout,
        )
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    rank = dist.get_rank(group=context.group)
"""

GDN_BWD_OLD = """    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)
"""

GDN_BWD_NEW = """    if context is None or context.group is None:
        return dht, initial_state
    if _adp_is_virtual_cp(context):
        # ADP patch: native virtual-chunk GDN backward state exchange.
        return _adp_chunk_gated_delta_rule_bwd_dhu_pre_process_virtual(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            bg=bg,
            scale=scale,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            dht=dht,
            initial_state=initial_state,
            context=context,
            transpose_state_layout=transpose_state_layout,
        )
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)
"""

GDN_COMPRESS_OLD = """def compress_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
"""

GDN_COMPRESS_NEW = """def compress_h0(h0: torch.Tensor, context: FLACPContext):
    if _adp_is_virtual_cp(context):
        # ADP patch: preserve native virtual-chunk initial states.
        return h0
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
"""

GDN_EXPAND_OLD = """def expand_h0(h0: torch.Tensor, context: FLACPContext):
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
"""

GDN_EXPAND_NEW = """def expand_h0(h0: torch.Tensor, context: FLACPContext):
    if _adp_is_virtual_cp(context):
        # ADP patch: preserve native virtual-chunk expanded states.
        return h0
    if h0 is None or len(context.cu_seqlens) == 2:
        return h0
"""

GDN_FWD_MULTISEQ_OLD = """    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    for local_idx in range(n_chunks):
        start = local_idx * chunk_len
        pre_process_fwd_kernel_merged[grid](
            k=_adp_narrow_seq(k, start, chunk_len),
            v=_adp_narrow_seq(u if v is None else v, start, chunk_len),
            w=_adp_narrow_seq(w, start, chunk_len),
            g=_adp_narrow_seq(g, start, chunk_len),
            gk=_adp_narrow_seq(gk, start, chunk_len),
            bg=_adp_narrow_seq(bg, start, chunk_len),
            u=_adp_narrow_seq(u, start, chunk_len),
            hm=summaries[local_idx],
            cu_seqlens=None,
            T=chunk_len,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=chunk_size,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
            MULTI_SEQS=False,
        )
"""

GDN_FWD_MULTISEQ_NEW = """    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, n_chunks)
    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u if v is None else v,
        w=w,
        g=g,
        gk=gk,
        bg=bg,
        u=u,
        hm=summaries,
        cu_seqlens=None,
        T=chunk_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BK1=BK,
        USE_EXP2=use_exp2,
        BLOCK_SIZE=BLOCK_SIZE,
        MULTI_SEQS=True,
    )
    # ADP patch: fuse native virtual GDN forward summaries.
"""

GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_OLD = """    USE_BG: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    \"\"\"
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)
    \"\"\"
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = 0
"""

GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_NEW = """    USE_BG: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
):
    \"\"\"
    Merged backward kernel that computes both dh (K x V) and dm (K x K) in a single kernel.

    Similar to pre_process_fwd_kernel_merged, this kernel uses a unified grid where:
    - Columns [0, V) are for computing dh (stage 1)
    - Columns [V, V+K) are for computing dm (stage 2)
    \"\"\"
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    if MULTI_SEQS:
        i_n = tl.program_id(2)
    else:
        i_n = 0
    # ADP patch: add multi-sequence native virtual GDN backward summary kernel support.
"""

GDN_BWD_KERNEL_MULTISEQ_OFFSET_OLD = """    dhm += i_h * K * (V + K)
    stride_qk = H * K
"""

GDN_BWD_KERNEL_MULTISEQ_OFFSET_NEW = """    dhm += (i_n * HV + i_h) * K * (V + K)
    # ADP patch: write native virtual GDN backward summaries per sequence.
    stride_qk = H * K
"""

GDN_BWD_MULTISEQ_OLD = """    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV)
    for local_idx in range(n_chunks):
        start = local_idx * chunk_len
        pre_process_bwd_kernel_merged[grid](
            q=_adp_narrow_seq(q, start, chunk_len),
            k=_adp_narrow_seq(k if bg is None else bg, start, chunk_len),
            w=_adp_narrow_seq(w, start, chunk_len),
            g=_adp_narrow_seq(g, start, chunk_len),
            gk=_adp_narrow_seq(gk, start, chunk_len),
            do=_adp_narrow_seq(do, start, chunk_len),
            dhm=summaries[local_idx],
            dv=_adp_narrow_seq(dv, start, chunk_len),
            cu_seqlens=None,
            scale=scale,
            T=chunk_len,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_BG=bg is not None,
        )
"""

GDN_BWD_MULTISEQ_NEW = """    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, n_chunks)
    pre_process_bwd_kernel_merged[grid](
        q=q,
        k=k if bg is None else bg,
        w=w,
        g=g,
        gk=gk,
        do=do,
        dhm=summaries,
        dv=dv,
        cu_seqlens=None,
        scale=scale,
        T=chunk_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK1=BK,
        USE_EXP2=use_exp2,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_BG=bg is not None,
        MULTI_SEQS=True,
    )
    # ADP patch: fuse native virtual GDN backward summaries.
"""

GDN_BWD_ORIGINAL_CALL_OLD = """            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_BG=bg is not None,
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
"""

GDN_BWD_ORIGINAL_CALL_NEW = """            BK1=BK,
            USE_EXP2=use_exp2,
            BLOCK_SIZE=BLOCK_SIZE,
            USE_BG=bg is not None,
            MULTI_SEQS=False,
        )

    # ADP patch: pass single-sequence flag to original GDN backward summary.
    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)
"""


def main() -> int:
    return max(
        patch_module(
            "megatron.core.ssm.gated_delta_net",
            MEGATRON_IMPORT_OLD,
            MEGATRON_IMPORT_NEW,
            MEGATRON_IMPORT_MARKER,
            "Megatron GDN native virtual CP import",
            missing_ok=True,
        ),
        patch_module(
            "megatron.core.ssm.gated_delta_net",
            MEGATRON_CONTEXT_OLD,
            MEGATRON_CONTEXT_NEW,
            MEGATRON_CONTEXT_MARKER,
            "Megatron GDN native virtual CP context",
            missing_ok=True,
        ),
        patch_module(
            "megatron.core.ssm.gated_delta_net",
            MEGATRON_UNDO_OLD,
            MEGATRON_UNDO_NEW,
            MEGATRON_SKIP_INPUT_EXCHANGE_MARKER,
            "Megatron GDN native virtual CP skip input exchange",
            missing_ok=True,
        ),
        patch_module(
            "megatron.core.ssm.gated_delta_net",
            MEGATRON_REDO_OLD,
            MEGATRON_REDO_NEW,
            MEGATRON_SKIP_OUTPUT_EXCHANGE_MARKER,
            "Megatron GDN native virtual CP skip output exchange",
            missing_ok=True,
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_HELPERS_OLD,
            CONV_HELPERS_NEW,
            CONV_HELPERS_MARKER,
            "FLA native virtual CP conv helpers",
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_PREP_OLD,
            CONV_PREP_NEW,
            CONV_PREP_MARKER,
            "FLA native virtual CP conv initial-state prep",
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_SIGNATURE_OLD,
            CONV_SIGNATURE_NEW,
            CONV_SIGNATURE_MARKER,
            "FLA native virtual CP conv correction signature",
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_CORRECT_OLD,
            CONV_CORRECT_NEW,
            CONV_CORRECT_MARKER,
            "FLA native virtual CP conv dx correction",
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_CTX_OLD,
            CONV_CTX_NEW,
            CONV_CTX_MARKER,
            "FLA native virtual CP conv context metadata",
        ),
        patch_module(
            "fla.modules.conv.cp.ops",
            CONV_BWD_OLD,
            CONV_BWD_NEW,
            CONV_BWD_CALL_MARKER,
            "FLA native virtual CP conv backward call",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_HELPERS_OLD,
            GDN_HELPERS_NEW,
            GDN_HELPERS_MARKER,
            "FLA native virtual CP GDN helpers",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_FWD_OLD,
            GDN_FWD_NEW,
            GDN_FWD_MARKER,
            "FLA native virtual CP GDN forward state exchange",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_BWD_OLD,
            GDN_BWD_NEW,
            GDN_BWD_MARKER,
            "FLA native virtual CP GDN backward state exchange",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_COMPRESS_OLD,
            GDN_COMPRESS_NEW,
            GDN_COMPRESS_MARKER,
            "FLA native virtual CP GDN h0 compression",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_EXPAND_OLD,
            GDN_EXPAND_NEW,
            GDN_EXPAND_MARKER,
            "FLA native virtual CP GDN h0 expansion",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_FWD_MULTISEQ_OLD,
            GDN_FWD_MULTISEQ_NEW,
            GDN_FWD_MULTISEQ_MARKER,
            "FLA native virtual CP GDN forward multi-sequence summaries",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_OLD,
            GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_NEW,
            GDN_BWD_KERNEL_MULTISEQ_SIGNATURE_MARKER,
            "FLA native virtual CP GDN backward multi-sequence kernel signature",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_BWD_KERNEL_MULTISEQ_OFFSET_OLD,
            GDN_BWD_KERNEL_MULTISEQ_OFFSET_NEW,
            GDN_BWD_KERNEL_MULTISEQ_OFFSET_MARKER,
            "FLA native virtual CP GDN backward multi-sequence summary offset",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_BWD_MULTISEQ_OLD,
            GDN_BWD_MULTISEQ_NEW,
            GDN_BWD_MULTISEQ_MARKER,
            "FLA native virtual CP GDN backward multi-sequence summaries",
        ),
        patch_module(
            "fla.ops.cp.chunk_delta_h",
            GDN_BWD_ORIGINAL_CALL_OLD,
            GDN_BWD_ORIGINAL_CALL_NEW,
            GDN_BWD_ORIGINAL_CALL_MARKER,
            "FLA original CP GDN backward single-sequence summary flag",
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
