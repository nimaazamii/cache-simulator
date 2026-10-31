"""Set-associative cache simulator (direct-mapped, n-way, fully associative).

Write-back with write-allocate. Replacement: LRU, FIFO or RANDOM (seeded).
"""
from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheStats:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writebacks: int = 0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.accesses if self.accesses else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


class Cache:
    """``associativity`` = 1 gives a direct-mapped cache; 0 (or None) gives a fully associative one."""

    POLICIES = ("LRU", "FIFO", "RANDOM")

    def __init__(self, size_bytes: int, block_bytes: int, associativity: int | None = 1,
                 policy: str = "LRU", seed: int = 0) -> None:
        if not (_is_pow2(size_bytes) and _is_pow2(block_bytes)):
            raise ValueError("size_bytes and block_bytes must be powers of two")
        if block_bytes > size_bytes:
            raise ValueError("block_bytes cannot exceed size_bytes")
        policy = policy.upper()
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {self.POLICIES}")
        num_blocks = size_bytes // block_bytes
        ways = num_blocks if not associativity else associativity
        if ways < 1 or ways > num_blocks or num_blocks % ways:
            raise ValueError("associativity must divide the number of blocks")
        self.size, self.block, self.ways, self.policy = size_bytes, block_bytes, ways, policy
        self.num_sets = num_blocks // ways
        self._sets: list[OrderedDict[int, bool]] = [OrderedDict() for _ in range(self.num_sets)]
        self._rng = random.Random(seed)
        self.stats = CacheStats()

    def access(self, address: int, write: bool = False) -> bool:
        """Access one byte address; returns True on a hit."""
        if address < 0:
            raise ValueError("address must be non-negative")
        block_addr = address // self.block
        tag, index = divmod(block_addr, self.num_sets)
        lines = self._sets[index]
        self.stats.accesses += 1
        if tag in lines:
            self.stats.hits += 1
            if self.policy == "LRU":
                lines.move_to_end(tag)
            if write:
                lines[tag] = True
            return True
        self.stats.misses += 1
        if len(lines) >= self.ways:
            victim = self._rng.choice(list(lines)) if self.policy == "RANDOM" else next(iter(lines))
            if lines.pop(victim):
                self.stats.writebacks += 1
            self.stats.evictions += 1
        lines[tag] = write
        return False

    def run(self, trace, writes: bool = False) -> CacheStats:
        for addr in trace:
            self.access(addr, write=writes)
        return self.stats


def classify_misses(trace, size_bytes: int, block_bytes: int, associativity: int | None = 1,
                    policy: str = "LRU") -> dict[str, int]:
    """Hill's 3C model: compulsory, capacity (also misses in a same-size fully associative LRU cache) and
    conflict (hits in that fully associative cache) misses."""
    real = Cache(size_bytes, block_bytes, associativity, policy)
    ideal = Cache(size_bytes, block_bytes, 0, "LRU")
    seen: set[int] = set()
    out = {"hits": 0, "compulsory": 0, "capacity": 0, "conflict": 0}
    for addr in trace:
        hit_real, hit_ideal = real.access(addr), ideal.access(addr)
        block = addr // block_bytes
        first = block not in seen
        seen.add(block)
        if hit_real:
            out["hits"] += 1
        elif first:
            out["compulsory"] += 1
        elif hit_ideal:
            out["conflict"] += 1
        else:
            out["capacity"] += 1
    return out


def amat(t_l1: float, miss_l1: float, t_l2: float, miss_l2_local: float, t_mem: float) -> float:
    """Average memory access time of a two-level hierarchy (local L2 miss rate)."""
    return t_l1 + miss_l1 * (t_l2 + miss_l2_local * t_mem)


class TwoLevelHierarchy:
    """L1 backed by L2 backed by memory; L2 only sees L1 misses."""

    def __init__(self, l1: Cache, l2: Cache, t_l1: float = 1.0, t_l2: float = 10.0, t_mem: float = 100.0) -> None:
        self.l1, self.l2 = l1, l2
        self.t_l1, self.t_l2, self.t_mem = t_l1, t_l2, t_mem

    def access(self, address: int, write: bool = False) -> None:
        if not self.l1.access(address, write):
            self.l2.access(address, write)

    def run(self, trace) -> float:
        for a in trace:
            self.access(a)
        return self.amat()

    def amat(self) -> float:
        return amat(self.t_l1, self.l1.stats.miss_rate, self.t_l2, self.l2.stats.miss_rate, self.t_mem)
