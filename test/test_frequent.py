# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

import unittest
from typing import Any

import networkx as nx
import pandas as pd

from fstg_toolkit.frequent.patterns import (
    FrequentPattern,
    FrequentPatterns,
    FrequentPatternsPopulationAnalysis,
    PatternEquivalenceStrategyRegistry,
    PatternStructure,
    PatternStructureTransitions,
    PatternStructureRegionsTransitions,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_pattern(nodes_regions: dict[int, str], edges: list[Any]) -> FrequentPattern:
    """Build a minimal FrequentPattern from node/region mappings and edge descriptors.

    Parameters
    ----------
    nodes_regions : dict[int, str]
        Mapping of node id to region label.
    edges : list
        Each element is either ``(src, dst)`` for a spatial edge or
        ``(src, dst, attrs_dict)`` for an edge with extra attributes
        (e.g. ``{'transition': 'PP'}``).

    Returns
    -------
    FrequentPattern
        The constructed pattern.
    """
    g = nx.DiGraph()
    for node_id, region in nodes_regions.items():
        g.add_node(node_id, region=region)
    for edge in edges:
        if len(edge) == 2:
            g.add_edge(edge[0], edge[1])
        else:
            g.add_edge(edge[0], edge[1], **edge[2])
    return FrequentPattern(g)


# ---------------------------------------------------------------------------
# Population fixtures
# ---------------------------------------------------------------------------

def _fixture_a() -> FrequentPatternsPopulationAnalysis:
    """No-factors population.

    Three subjects, Subject index only.

    Pattern A  (R1→R2, spatial) — shared by S1 and S2.
    Pattern B  (R1→R2, spatial, structurally identical to A) — S2 duplicate.
    Pattern C  (R1→R3, spatial) — shared by S2 and S3.

    Expected: unique_patterns = [A, C] (len 2), 4-row track.
    """
    p_a = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
    p_b = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])   # equivalent to A
    p_c = _make_pattern({0: 'R1', 1: 'R3'}, [(0, 1)])
    patterns = {
        ("S1",): FrequentPatterns(patterns={"p1": p_a}),
        ("S2",): FrequentPatterns(patterns={"p1": p_b, "p2": p_c}),
        ("S3",): FrequentPatterns(patterns={"p1": p_c}),
    }
    return FrequentPatternsPopulationAnalysis(patterns, ("Subject",), PatternStructureRegionsTransitions)


def _fixture_b() -> FrequentPatternsPopulationAnalysis:
    """With-factors population.

    Three (Subject, Session) entries.

    (S1, Ses1) → Pattern A  (R1→R2, spatial)
    (S2, Ses1) → Pattern C  (R1→R3, spatial)
    (S1, Ses2) → Pattern C

    Expected: unique_patterns = [A, C] (len 2), 3-row track.
    """
    p_a = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
    p_c = _make_pattern({0: 'R1', 1: 'R3'}, [(0, 1)])
    patterns = {
        ("S1", "Ses1"): FrequentPatterns(patterns={"p1": p_a}),
        ("S2", "Ses1"): FrequentPatterns(patterns={"p1": p_c}),
        ("S1", "Ses2"): FrequentPatterns(patterns={"p1": p_c}),
    }
    return FrequentPatternsPopulationAnalysis(
        patterns, ("Subject", "Session"), PatternStructureRegionsTransitions
    )


def _fixture_temporal_a() -> FrequentPatternsPopulationAnalysis:
    """No-factors population with both spatial and temporal edges.

    S1 has Pattern A (spatial R1→R2) and Pattern E (temporal R1→R1, transition PP).
    """
    p_a = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
    p_e = _make_pattern({0: 'R1', 1: 'R1'}, [(0, 1, {'transition': 'PP'})])
    patterns = {
        ("S1",): FrequentPatterns(patterns={"p1": p_a, "p2": p_e}),
    }
    return FrequentPatternsPopulationAnalysis(patterns, ("Subject",), PatternStructureRegionsTransitions)


def _fixture_temporal_b() -> FrequentPatternsPopulationAnalysis:
    """With-factors population containing temporal edges.

    (S1, Ses1) → Pattern E (temporal)
    (S2, Ses1) → Pattern A (spatial)
    (S1, Ses2) → Pattern E (temporal)
    """
    p_a = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
    p_e = _make_pattern({0: 'R1', 1: 'R1'}, [(0, 1, {'transition': 'PP'})])
    patterns = {
        ("S1", "Ses1"): FrequentPatterns(patterns={"p1": p_e}),
        ("S2", "Ses1"): FrequentPatterns(patterns={"p1": p_a}),
        ("S1", "Ses2"): FrequentPatterns(patterns={"p1": p_e}),
    }
    return FrequentPatternsPopulationAnalysis(
        patterns, ("Subject", "Session"), PatternStructureRegionsTransitions
    )


# ===========================================================================
# FrequentPattern
# ===========================================================================

class FrequentPatternFromDictTestCase(unittest.TestCase):
    def test_nodes_created_with_attributes(self):
        d = {
            'nodes': [{'id': 0, 'region': 'R1'}, {'id': 1, 'region': 'R2'}],
            'edges': [],
        }
        p = FrequentPattern.from_dict(d)
        self.assertEqual(set(p.nodes()), {0, 1})
        self.assertEqual(p.nodes[0]['region'], 'R1')
        self.assertEqual(p.nodes[1]['region'], 'R2')

    def test_edges_created_with_attributes(self):
        d = {
            'nodes': [{'id': 0, 'region': 'R1'}, {'id': 1, 'region': 'R2'}],
            'edges': [{'source': 0, 'target': 1, 'transition': 'PP'}],
        }
        p = FrequentPattern.from_dict(d)
        self.assertIn((0, 1), p.edges())
        self.assertEqual(p[0][1]['transition'], 'PP')


# ===========================================================================
# PatternEquivalenceStrategyRegistry
# ===========================================================================

class PatternEquivalenceStrategyRegistryTestCase(unittest.TestCase):
    def test_names_returns_sorted_list(self):
        names = PatternEquivalenceStrategyRegistry.names()
        self.assertIn('structure', names)
        self.assertIn('structure-transitions', names)
        self.assertIn('structure-regions-transitions', names)
        self.assertEqual(names, sorted(names))

    def test_get_returns_registered_class(self):
        self.assertIs(PatternEquivalenceStrategyRegistry.get('structure'), PatternStructure)

    def test_get_unknown_raises_key_error(self):
        with self.assertRaises(KeyError):
            PatternEquivalenceStrategyRegistry.get('nonexistent-strategy')


# ===========================================================================
# Equivalence strategies
# ===========================================================================

class PatternStructureEquivalenceTestCase(unittest.TestCase):
    def test_isomorphic_graphs_are_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        p2 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        self.assertTrue(PatternStructure.equivalent(p1, p2))

    def test_non_isomorphic_not_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        p2 = _make_pattern({0: 'R1', 1: 'R2', 2: 'R3'}, [(0, 1), (1, 2)])
        self.assertFalse(PatternStructure.equivalent(p1, p2))


class PatternStructureTransitionsEquivalenceTestCase(unittest.TestCase):
    def test_same_transitions_are_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PP'})])
        p2 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PP'})])
        self.assertTrue(PatternStructureTransitions.equivalent(p1, p2))

    def test_different_transitions_not_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PP'})])
        p2 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PO'})])
        self.assertFalse(PatternStructureTransitions.equivalent(p1, p2))

    def test_non_isomorphic_not_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PP'})])
        p2 = _make_pattern(
            {0: 'R1', 1: 'R2', 2: 'R3'},
            [(0, 1, {'transition': 'PP'}), (1, 2, {'transition': 'PP'})],
        )
        self.assertFalse(PatternStructureTransitions.equivalent(p1, p2))


class PatternStructureRegionsTransitionsEquivalenceTestCase(unittest.TestCase):
    def test_identical_patterns_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        p2 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        self.assertTrue(PatternStructureRegionsTransitions.equivalent(p1, p2))

    def test_different_region_not_equivalent(self):
        p1 = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        p2 = _make_pattern({0: 'R1', 1: 'R3'}, [(0, 1)])
        self.assertFalse(PatternStructureRegionsTransitions.equivalent(p1, p2))


# ===========================================================================
# Static helpers
# ===========================================================================

class IterCountsTestCase(unittest.TestCase):
    def test_without_factors_yields_empty_dict(self):
        df = pd.DataFrame({'Count': [3, 5]}, index=pd.Index([0, 1], name='idx'))
        result = list(FrequentPatternsPopulationAnalysis._iter_counts(df, []))
        self.assertEqual(result, [({}, 0, 3), ({}, 1, 5)])

    def test_with_factors_yields_factor_dict(self):
        idx = pd.MultiIndex.from_tuples([('Ses1', 0), ('Ses2', 1)], names=['Session', 'idx'])
        df = pd.DataFrame({'Count': [2, 4]}, index=idx)
        result = list(FrequentPatternsPopulationAnalysis._iter_counts(df, ['Session']))
        self.assertEqual(result, [({'Session': 'Ses1'}, 0, 2), ({'Session': 'Ses2'}, 1, 4)])


class CollectAllRegionsTestCase(unittest.TestCase):
    def test_returns_sorted_labels_and_index(self):
        p1 = _make_pattern({0: 'R2', 1: 'R1'}, [])
        p2 = _make_pattern({0: 'R3'}, [])
        labels, idx = FrequentPatternsPopulationAnalysis._collect_all_regions([p1, p2])
        self.assertEqual(labels, ['R1', 'R2', 'R3'])
        self.assertEqual(idx, {'R1': 0, 'R2': 1, 'R3': 2})

    def test_empty_list_returns_empty(self):
        labels, idx = FrequentPatternsPopulationAnalysis._collect_all_regions([])
        self.assertEqual(labels, [])
        self.assertEqual(idx, {})


class CountSpatialEdgePairsTestCase(unittest.TestCase):
    def test_spatial_edges_produce_pairs(self):
        p = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1)])
        pairs = FrequentPatternsPopulationAnalysis._count_spatial_edge_pairs(p, 3)
        self.assertEqual(pairs, {('R1', 'R2'): 3})

    def test_transition_edges_skipped(self):
        p = _make_pattern({0: 'R1', 1: 'R2'}, [(0, 1, {'transition': 'PP'})])
        pairs = FrequentPatternsPopulationAnalysis._count_spatial_edge_pairs(p, 1)
        self.assertEqual(pairs, {})

    def test_same_region_edges_skipped(self):
        p = _make_pattern({0: 'R1', 1: 'R1'}, [(0, 1)])
        pairs = FrequentPatternsPopulationAnalysis._count_spatial_edge_pairs(p, 1)
        self.assertEqual(pairs, {})

    def test_pair_is_sorted(self):
        # Edge goes from R2→R1, but the stored pair should be sorted ('R1', 'R2')
        p = _make_pattern({0: 'R2', 1: 'R1'}, [(0, 1)])
        pairs = FrequentPatternsPopulationAnalysis._count_spatial_edge_pairs(p, 1)
        self.assertIn(('R1', 'R2'), pairs)


class BuildSymmetricMatrixTestCase(unittest.TestCase):
    def test_symmetric_placement(self):
        pairs = {('R1', 'R2'): 3}
        region_idx = {'R1': 0, 'R2': 1, 'R3': 2}
        matrix = FrequentPatternsPopulationAnalysis._build_symmetric_matrix(pairs, region_idx, 3)
        self.assertEqual(matrix[0][1], 3)
        self.assertEqual(matrix[1][0], 3)
        self.assertEqual(matrix[0][0], 0)

    def test_empty_pairs_zero_matrix(self):
        matrix = FrequentPatternsPopulationAnalysis._build_symmetric_matrix({}, {}, 2)
        self.assertEqual(matrix, [[0, 0], [0, 0]])


class IncrementPatternMatrixTestCase(unittest.TestCase):
    def test_diagonal_incremented_for_each_index(self):
        matrix = [[0, 0], [0, 0]]
        FrequentPatternsPopulationAnalysis._increment_pattern_matrix(matrix, [0, 1])
        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[1][1], 1)

    def test_off_diagonal_incremented_for_pairs(self):
        matrix = [[0, 0], [0, 0]]
        FrequentPatternsPopulationAnalysis._increment_pattern_matrix(matrix, [0, 1])
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[1][0], 1)

    def test_single_index_only_diagonal(self):
        matrix = [[0, 0], [0, 0]]
        FrequentPatternsPopulationAnalysis._increment_pattern_matrix(matrix, [0])
        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[0][1], 0)
        self.assertEqual(matrix[1][0], 0)


# ===========================================================================
# FrequentPatternsPopulationAnalysis — constructor
# ===========================================================================

class FrequentPatternsPopulationAnalysisConstructorTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis = _fixture_a()

    def test_deduplicates_equivalent_patterns(self):
        # B is equivalent to A; unique patterns should be [A, C]
        self.assertEqual(len(self.analysis.unique_patterns), 2)

    def test_track_has_correct_row_count(self):
        # S1→1 row, S2→2 rows (A+C), S3→1 row
        self.assertEqual(len(self.analysis.track), 4)

    def test_track_indices_correct(self):
        track = self.analysis.track
        s1_idx = sorted(track[track.index == 'S1']['idx'].tolist())
        s2_idx = sorted(track[track.index == 'S2']['idx'].tolist())
        s3_idx = sorted(track[track.index == 'S3']['idx'].tolist())
        self.assertEqual(s1_idx, [0])
        self.assertEqual(s2_idx, [0, 1])
        self.assertEqual(s3_idx, [1])


# ===========================================================================
# get_counts
# ===========================================================================

class GetCountsTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_correct_counts(self):
        counts = self.analysis_a.get_counts([])
        # pattern 0 (A) appears in S1 and S2 → count 2
        # pattern 1 (C) appears in S2 and S3 → count 2
        self.assertEqual(counts.loc[0, 'Count'], 2)
        self.assertEqual(counts.loc[1, 'Count'], 2)

    def test_with_factors_correct_counts(self):
        counts = self.analysis_b.get_counts(['Session'])
        self.assertEqual(counts.loc[('Ses1', 0), 'Count'], 1)
        self.assertEqual(counts.loc[('Ses1', 1), 'Count'], 1)
        self.assertEqual(counts.loc[('Ses2', 1), 'Count'], 1)


# ===========================================================================
# get_patterns_per_region
# ===========================================================================

class GetPatternsPerRegionTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_all_regions_present(self):
        result = self.analysis_a.get_patterns_per_region([])
        regions = result['Region'].values
        self.assertIn('R1', regions)
        self.assertIn('R2', regions)
        self.assertIn('R3', regions)

    def test_without_factors_r1_aggregated_count(self):
        # R1 appears in both patterns (each with subject count 2): total = 4
        result = self.analysis_a.get_patterns_per_region([])
        r1_count = result.loc[result['Region'] == 'R1', 'Count'].values[0]
        self.assertEqual(r1_count, 4)

    def test_without_factors_pattern_indices_column(self):
        result = self.analysis_a.get_patterns_per_region([])
        self.assertIn('PatternIndices', result.columns)

    def test_with_factors_session_column_present(self):
        result = self.analysis_b.get_patterns_per_region(['Session'])
        self.assertIn('Session', result.columns)
        self.assertIn('Ses1', result['Session'].values)
        self.assertIn('Ses2', result['Session'].values)


# ===========================================================================
# get_temporal_dynamics
# ===========================================================================

class GetTemporalDynamicsTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_temporal_a()
        self.analysis_b = _fixture_temporal_b()

    def test_temporal_edges_captured(self):
        result = self.analysis_a.get_temporal_dynamics([])
        self.assertIn('R1', result['Region'].values)
        self.assertIn('PP', result['Transition'].values)

    def test_spatial_edges_excluded(self):
        # R2 only appears via a spatial edge in pattern A; must not appear in temporal dynamics
        result = self.analysis_a.get_temporal_dynamics([])
        self.assertNotIn('R2', result['Region'].values)

    def test_with_factors_session_column_present(self):
        result = self.analysis_b.get_temporal_dynamics(['Session'])
        self.assertIn('Session', result.columns)
        self.assertIn('Ses1', result['Session'].values)
        self.assertIn('Ses2', result['Session'].values)


# ===========================================================================
# get_region_co_occurrence
# ===========================================================================

class GetRegionCoOccurrenceTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_key_is_empty_tuple(self):
        result = self.analysis_a.get_region_co_occurrence([])
        self.assertIn((), result)

    def test_returns_labels_and_matrix(self):
        result = self.analysis_a.get_region_co_occurrence([])
        labels, matrix = result[()]
        self.assertIsInstance(labels, list)
        self.assertIsInstance(matrix, list)
        self.assertEqual(len(matrix), len(labels))

    def test_matrix_is_symmetric(self):
        result = self.analysis_a.get_region_co_occurrence([])
        _, matrix = result[()]
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                self.assertEqual(matrix[i][j], matrix[j][i])

    def test_r1_r2_co_occurrence_count(self):
        # Pattern A (R1-R2 spatial edge) has subject count 2 → co-occurrence = 2
        result = self.analysis_a.get_region_co_occurrence([])
        labels, matrix = result[()]
        r1, r2 = labels.index('R1'), labels.index('R2')
        self.assertEqual(matrix[r1][r2], 2)

    def test_with_factors_session_keys(self):
        result = self.analysis_b.get_region_co_occurrence(['Session'])
        self.assertIn(('Ses1',), result)
        self.assertIn(('Ses2',), result)


# ===========================================================================
# get_pattern_co_occurrence
# ===========================================================================

class GetPatternCoOccurrenceTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_key_is_empty_tuple(self):
        result = self.analysis_a.get_pattern_co_occurrence([])
        self.assertIn((), result)

    def test_diagonal_counts_pattern_occurrences(self):
        result = self.analysis_a.get_pattern_co_occurrence([])
        matrix = result[()]
        # pattern 0 (A): S1 and S2 → diagonal 2
        # pattern 1 (C): S2 and S3 → diagonal 2
        self.assertEqual(matrix[0][0], 2)
        self.assertEqual(matrix[1][1], 2)

    def test_off_diagonal_counts_cooccurrence(self):
        result = self.analysis_a.get_pattern_co_occurrence([])
        matrix = result[()]
        # only S2 has both patterns → off-diagonal 1
        self.assertEqual(matrix[0][1], 1)
        self.assertEqual(matrix[1][0], 1)

    def test_with_factors_session_keys(self):
        result = self.analysis_b.get_pattern_co_occurrence(['Session'])
        self.assertIn(('Ses1',), result)
        self.assertIn(('Ses2',), result)

    def test_with_factors_correct_diagonal(self):
        result = self.analysis_b.get_pattern_co_occurrence(['Session'])
        # Ses1: S1→A(idx0), S2→C(idx1) — each appears once, no co-occurrence
        matrix_ses1 = result[('Ses1',)]
        self.assertEqual(matrix_ses1[0][0], 1)
        self.assertEqual(matrix_ses1[1][1], 1)
        self.assertEqual(matrix_ses1[0][1], 0)


# ===========================================================================
# get_occurrence_histogram
# ===========================================================================

class GetOccurrenceHistogramTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_columns_present(self):
        result = self.analysis_a.get_occurrence_histogram([])
        self.assertIn('Occurrences', result.columns)
        self.assertIn('Patterns', result.columns)
        self.assertIn('PatternIndices', result.columns)

    def test_without_factors_bin_count(self):
        # Both patterns have subject count 2 → one bin: Occurrences=2, Patterns=2
        result = self.analysis_a.get_occurrence_histogram([])
        row = result[result['Occurrences'] == 2]
        self.assertFalse(row.empty)
        self.assertEqual(row['Patterns'].values[0], 2)

    def test_with_factors_session_column(self):
        result = self.analysis_b.get_occurrence_histogram(['Session'])
        self.assertIn('Session', result.columns)
        self.assertIn('Ses1', result['Session'].values)
        self.assertIn('Ses2', result['Session'].values)


# ===========================================================================
# get_pattern_complexity
# ===========================================================================

class GetPatternComplexityTestCase(unittest.TestCase):
    def setUp(self):
        self.analysis_a = _fixture_a()
        self.analysis_b = _fixture_b()

    def test_without_factors_columns_present(self):
        result = self.analysis_a.get_pattern_complexity([])
        self.assertIn('Size', result.columns)
        self.assertIn('Count', result.columns)
        self.assertIn('PatternIndices', result.columns)

    def test_without_factors_correct_size(self):
        # Both patterns A and C have 2 nodes
        result = self.analysis_a.get_pattern_complexity([])
        self.assertIn(2, result['Size'].values.tolist())

    def test_with_factors_session_column(self):
        result = self.analysis_b.get_pattern_complexity(['Session'])
        self.assertIn('Session', result.columns)
        self.assertIn('Ses1', result['Session'].values)
        self.assertIn('Ses2', result['Session'].values)


if __name__ == '__main__':
    unittest.main()
