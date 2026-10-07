"""Optional Triton micro-kernel for MLP inference building blocks.

This is intentionally a kernel-level experiment, not a replacement for the
PyTorch model. On supported GPUs it fuses bias addition and ReLU for a dense
activation tensor, reducing an intermediate read/write.
"""
from __future__ import annotations

def fused_bias_relu(x, bias):
    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Triton and PyTorch are required") from exc

    if not x.is_cuda or not bias.is_cuda:
        raise RuntimeError("fused_bias_relu requires CUDA tensors")
    if x.ndim != 2 or bias.ndim != 1 or x.shape[1] != bias.shape[0]:
        raise ValueError("expected x=[rows, features] and bias=[features]")

    @triton.jit
    def kernel(xp, bp, yp, n_rows, n_cols, stride_x0, stride_x1, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        row = pid // tl.cdiv(n_cols, BLOCK)
        col = (pid % tl.cdiv(n_cols, BLOCK)) * BLOCK + offs
        mask = (row < n_rows) & (col < n_cols)
        xval = tl.load(xp + row * stride_x0 + col * stride_x1, mask=mask, other=0.0)
        bval = tl.load(bp + col, mask=col < n_cols, other=0.0)
        tl.store(yp + row * stride_x0 + col * stride_x1, tl.maximum(xval + bval, 0.0), mask=mask)

    out=torch.empty_like(x)
    grid=lambda meta: (x.shape[0] * ((x.shape[1] + meta['BLOCK'] - 1)//meta['BLOCK']),)
    kernel[grid](x,bias,out,x.shape[0],x.shape[1],x.stride(0),x.stride(1),BLOCK=128)
    return out
