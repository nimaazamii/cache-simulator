"""Cache experiments on synthetic traces; writes results/experiments.json and a figure.

    python examples/run_experiments.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cachesim import Cache, TwoLevelHierarchy, classify_misses, traces

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    res: dict = {"DATA": "synthetic address traces (no real program traces)"}

    # 1. Associativity vs. conflict misses: 8 addresses 4 KiB apart, touched 50 times, 8 KiB cache, 64 B blocks
    trace = traces.strided_blocks(8, 4096, 50)
    assoc = {}
    for ways in (1, 2, 4, 8, 0):
        c = Cache(8192, 64, ways)
        assoc["fully" if ways == 0 else str(ways)] = c.run(trace).miss_rate
    res["associativity_vs_miss_rate_strided"] = {"trace_length": len(trace), "miss_rate": assoc}
    res["miss_classification_direct_mapped_strided"] = classify_misses(trace, 8192, 64, 1)

    # 2. Cache size vs. miss rate on random accesses over a 32 KiB space (4-way LRU, 32 B blocks)
    rnd = traces.random_access(50_000, 32768, seed=0)
    res["size_vs_miss_rate_random"] = {str(s): Cache(s, 32, 4).run(rnd).miss_rate
                                       for s in (1024, 2048, 4096, 8192, 16384, 32768)}

    # 3. Replacement policies on a loop over 10 blocks in an 8-block fully associative cache
    lp = traces.loop(10 * 64, repeats=100, stride=64)
    res["policy_loop_10_blocks_in_8"] = {p: Cache(512, 64, 0, p, seed=0).run(lp).hit_rate
                                         for p in ("LRU", "FIFO", "RANDOM")}

    # 4. Row-major vs column-major traversal of a 64x64 int matrix, 4 KiB 4-way cache with 64 B blocks
    res["matrix_64x64_miss_rate"] = {o: Cache(4096, 64, 4).run(traces.matrix_traversal(64, 64, o)).miss_rate
                                     for o in ("row", "col")}

    # 5. Two-level hierarchy AMAT (L1 4 KiB 2-way, L2 64 KiB 8-way, 64 B blocks; 1 / 10 / 100 cycles)
    amat_res = {}
    for name, tr in (("row-major", traces.matrix_traversal(256, 256, "row")),
                     ("column-major", traces.matrix_traversal(256, 256, "col"))):
        h = TwoLevelHierarchy(Cache(4096, 64, 2), Cache(65536, 64, 8))
        h.run(tr)
        amat_res[name] = {"amat_cycles": h.amat(), "l1_miss_rate": h.l1.stats.miss_rate,
                          "l2_local_miss_rate": h.l2.stats.miss_rate}
    res["two_level_amat_256x256_matrix"] = amat_res

    (OUT / "experiments.json").write_text(json.dumps(res, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(list(assoc), list(assoc.values())); ax[0].set(xlabel="associativity (ways)", ylabel="miss rate",
                                                             title="Strided trace, 8 KiB cache")
    sizes = res["size_vs_miss_rate_random"]
    ax[1].plot([int(s) / 1024 for s in sizes], list(sizes.values()), "o-")
    ax[1].set(xlabel="cache size (KiB)", ylabel="miss rate", title="Random accesses over 32 KiB", xscale="log", xticks=[1, 2, 4, 8, 16, 32], xticklabels=[1, 2, 4, 8, 16, 32])
    fig.tight_layout(); fig.savefig(OUT / "experiments.png", dpi=120)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
