import unittest

from cachesim import Cache, TwoLevelHierarchy, amat, classify_misses, traces


class TestMapping(unittest.TestCase):
    def test_direct_mapped_conflict(self):
        c = Cache(64, 16, 1)                    # 4 blocks, 4 sets
        for a in (0, 64, 0, 64):                # block 0 and block 4 share set 0
            c.access(a)
        self.assertEqual((c.stats.hits, c.stats.misses), (0, 4))

    def test_two_way_removes_that_conflict(self):
        c = Cache(64, 16, 2)                    # 2 sets of 2 ways
        for a in (0, 64, 0, 64):
            c.access(a)
        self.assertEqual((c.stats.hits, c.stats.misses), (2, 2))

    def test_spatial_locality_within_a_block(self):
        c = Cache(64, 16, 1)
        for a in (0, 4, 8, 12):
            c.access(a)
        self.assertEqual((c.stats.hits, c.stats.misses), (3, 1))

    def test_fully_associative_geometry(self):
        c = Cache(64, 16, 0)
        self.assertEqual((c.num_sets, c.ways), (1, 4))

    def test_stats_are_consistent(self):
        c = Cache(256, 16, 2)
        c.run(traces.random_access(500, 4096, seed=1))
        s = c.stats
        self.assertEqual(s.hits + s.misses, s.accesses)
        self.assertEqual(s.accesses, 500)


class TestReplacementPolicies(unittest.TestCase):
    TRACE = [0, 16, 0, 32, 16]   # A B A C B on a fully associative 2-block cache

    def _hits(self, policy):
        c = Cache(32, 16, 0, policy)
        c.run(self.TRACE)
        return c.stats.hits

    def test_lru_and_fifo_differ_on_the_classic_trace(self):
        self.assertEqual(self._hits("LRU"), 1)
        self.assertEqual(self._hits("FIFO"), 2)

    def test_random_is_reproducible_for_a_seed_and_valid(self):
        trace = traces.random_access(2000, 2048, seed=3)
        a, b = Cache(512, 16, 4, "RANDOM", seed=7), Cache(512, 16, 4, "RANDOM", seed=7)
        self.assertEqual(a.run(trace).hits, b.run(trace).hits)

    def test_lru_is_worst_case_for_loop_slightly_larger_than_cache(self):
        trace = traces.loop(10 * 16, repeats=20, stride=16)    # 10 blocks cycling through 8
        lru, rnd = Cache(128, 16, 0, "LRU"), Cache(128, 16, 0, "RANDOM", seed=0)
        self.assertEqual(lru.run(trace).hits, 0)
        self.assertGreater(rnd.run(trace).hits, 0)

    def test_dirty_eviction_counts_writeback(self):
        c = Cache(16, 16, 1)
        c.access(0, write=True)
        c.access(16)               # evicts dirty block 0
        c.access(32)               # evicts clean block 1
        self.assertEqual((c.stats.evictions, c.stats.writebacks), (2, 1))


class TestMissClassification(unittest.TestCase):
    def test_ping_pong_is_all_conflict_misses(self):
        out = classify_misses([0, 64] * 10, 64, 16, 1)
        self.assertEqual(out, {"hits": 0, "compulsory": 2, "capacity": 0, "conflict": 18})

    def test_scan_larger_than_cache_is_capacity_misses(self):
        trace = traces.sequential(8, stride=16) * 2    # 8 blocks twice through a 4-block cache
        out = classify_misses(trace, 64, 16, 0)
        self.assertEqual(out, {"hits": 0, "compulsory": 8, "capacity": 8, "conflict": 0})

    def test_categories_sum_to_trace_length(self):
        trace = traces.random_access(3000, 8192, seed=2)
        out = classify_misses(trace, 1024, 32, 2)
        self.assertEqual(sum(out.values()), 3000)


class TestHierarchyAndTraces(unittest.TestCase):
    def test_amat_formula(self):
        self.assertAlmostEqual(amat(1, 0.1, 10, 0.5, 100), 7.0)

    def test_l2_sees_only_l1_misses(self):
        h = TwoLevelHierarchy(Cache(256, 16, 1), Cache(2048, 16, 4))
        h.run(traces.random_access(4000, 8192, seed=5))
        self.assertEqual(h.l2.stats.accesses, h.l1.stats.misses)
        self.assertGreater(h.amat(), h.t_l1)

    def test_row_major_vs_column_major_matrix(self):
        row = Cache(4096, 64, 4).run(traces.matrix_traversal(64, 64, "row"))
        col = Cache(4096, 64, 4).run(traces.matrix_traversal(64, 64, "col"))
        self.assertAlmostEqual(row.miss_rate, 1 / 16)          # one miss per 16 ints
        self.assertGreater(col.miss_rate, 5 * row.miss_rate)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Cache(100, 16, 1)
        with self.assertRaises(ValueError):
            Cache(64, 16, 3)
        with self.assertRaises(ValueError):
            Cache(64, 16, 1, "MRU")
        with self.assertRaises(ValueError):
            Cache(64, 16, 1).access(-4)
        with self.assertRaises(ValueError):
            traces.matrix_traversal(2, 2, "diag")


if __name__ == "__main__":
    unittest.main()
