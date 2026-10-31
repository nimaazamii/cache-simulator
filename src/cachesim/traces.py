"""Synthetic memory-access trace generators (byte addresses)."""
from __future__ import annotations

import random


def sequential(n: int, stride: int = 4, base: int = 0) -> list[int]:
    return [base + i * stride for i in range(n)]


def loop(array_bytes: int, repeats: int, stride: int = 4, base: int = 0) -> list[int]:
    one_pass = list(range(base, base + array_bytes, stride))
    return one_pass * repeats


def strided_blocks(count: int, stride_bytes: int, repeats: int, base: int = 0) -> list[int]:
    """Touch ``count`` addresses ``stride_bytes`` apart, ``repeats`` times (power-of-two strides cause conflicts)."""
    return [base + i * stride_bytes for i in range(count)] * repeats


def random_access(n: int, space_bytes: int, seed: int = 0, align: int = 4) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, space_bytes // align) * align for _ in range(n)]


def matrix_traversal(rows: int, cols: int, order: str = "row", elem_bytes: int = 4, base: int = 0) -> list[int]:
    """Addresses of a row-major matrix visited row by row or column by column."""
    if order == "row":
        return [base + (r * cols + c) * elem_bytes for r in range(rows) for c in range(cols)]
    if order == "col":
        return [base + (r * cols + c) * elem_bytes for c in range(cols) for r in range(rows)]
    raise ValueError("order must be 'row' or 'col'")
