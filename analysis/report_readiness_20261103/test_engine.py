import importlib.util
from pathlib import Path
import unittest
import sys
import numpy as np


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('engine.py')
        if not path.exists():
            cls.engine = None
            return
        spec = importlib.util.spec_from_file_location('analysis_engine', path)
        cls.engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.engine)

    def setUp(self):
        self.assertIsNotNone(self.engine, 'Analysis engine not implemented yet')

    def test_weighted_isotonic_pools_decreasing_blocks(self):
        fit = self.engine.fit_iso(np.array([0., 0., 1., 2.]), np.array([1, 1, 0, 1]))
        np.testing.assert_allclose(self.engine.apply_iso(np.array([0., 1., 2.]), fit), [2/3, 2/3, 1])

    def test_missing_confidence_escalates_at_zero_cutoff(self):
        fit = self.engine.fit_iso(np.array([0., 1.]), np.array([0, 1]))
        mapped = self.engine.apply_iso(np.array([np.nan, 0., 1.]), fit)
        np.testing.assert_array_equal(mapped >= 0., [False, True, True])

    def test_all_wrong_and_all_right_calibration(self):
        for y in [np.zeros(4), np.ones(4)]:
            f = self.engine.fit_iso(np.arange(4.), y)
            np.testing.assert_allclose(self.engine.apply_iso(np.array([-10., 1., 10.]), f), y[0])

    def test_selection_uses_strong_baseline_not_sweep_maximum(self):
        ycheap = np.array([0, 0, 1, 1]); ystrong = np.array([1, 1, 0, 0])
        scores = {'token': np.array([.1, .1, .9, .9])}
        r = self.engine.select_policy(scores, ycheap, ystrong, np.ones(4)*.1, np.ones(4), np.array([0., .5, 1.000001]))
        self.assertEqual(r['threshold'], 0.)
        self.assertEqual(r['correct'], 2)
        self.assertAlmostEqual(r['cost'], .4)

    def test_cheapest_qualifying_rule_can_cost_more_than_strong_only(self):
        r = self.engine.select_policy({'token':np.zeros(2)}, np.zeros(2), np.ones(2), np.ones(2), np.ones(2), np.array([0., 1.000001]))
        self.assertEqual(r['threshold'], 1.000001)
        self.assertEqual(r['cost'], 4.)

    def test_zero_strong_correct_has_no_defined_retention(self):
        r = self.engine.evaluate(np.zeros(2), np.zeros(2), np.ones(2), np.ones(2), np.array([True, False]))
        self.assertIsNone(r['retention'])
        self.assertEqual(r['correct'], 0)

    def test_expected_cost_matching_accounts_for_heterogeneous_requests(self):
        cheap = np.array([.2,.2,.2]); strong=np.array([1.,3.,6.]); keep=np.array([True,True,False])
        p=self.engine.budget_probability(cheap,strong,keep)
        self.assertAlmostEqual(p,.6)
        self.assertAlmostEqual(cheap.sum()+p*strong.sum(),cheap.sum()+strong[~keep].sum())

    def test_question_groups_join_id_or_text_transitively(self):
        groups=self.engine.question_groups([{'question_id':'a','question':'X'}, {'question_id':'a','question':'Y'}, {'question_id':'b','question':' y '}, {'question_id':'','question':'Z'}])
        self.assertEqual(groups[0],groups[1]); self.assertEqual(groups[1],groups[2]); self.assertNotEqual(groups[2],groups[3])

    def test_group_folds_never_split_a_question(self):
        groups=np.array(['a','a','b','c','d','d','e','f']); source=np.array([0,1,0,1,0,1,0,1])
        folds=self.engine.group_folds(groups,source,3,42)
        self.assertEqual(set(folds),{0,1,2})
        for g in set(groups):self.assertEqual(len(set(folds[groups==g])),1)
        np.testing.assert_array_equal(folds,self.engine.group_folds(groups,source,3,42))

    def test_crossfit_scores_do_not_use_held_group_labels(self):
        x=np.arange(9.)/8; y=np.array([0,1,0,1,0,1,0,1,0]); f=np.arange(9)%3
        first=self.engine.crossfit_iso(x,y,f)
        changed=y.copy(); changed[f==0]=1-changed[f==0]
        second=self.engine.crossfit_iso(x,changed,f)
        np.testing.assert_allclose(first[f==0],second[f==0])

    def test_cluster_bootstrap_keeps_within_group_cancellation(self):
        samples=self.engine.cluster_bootstrap_mean(np.array([1.,-1.,2.,-2.]), np.array(['a','a','b','b']),100,42)
        np.testing.assert_array_equal(samples,np.zeros(100))

    def test_random_expected_correctness_at_probability_endpoints(self):
        a=np.array([1.,0.,0.]); b=np.array([0.,1.,1.])
        self.assertEqual(self.engine.random_expected_correct(a,b,0),1.)
        self.assertEqual(self.engine.random_expected_correct(a,b,1),2.)

    def test_budget_matched_bootstrap_keeps_whole_question_groups(self):
        self.assertTrue(hasattr(self.engine,'bootstrap_control'),'Budget bootstrap not implemented yet')
        a=np.array([1,0,1,0]);b=1-a;keep=np.array([True,False,True,False])
        result=self.engine.bootstrap_control(a,b,np.ones(4)*.1,np.ones(4),keep,np.array(['a','a','b','b']),100,42)
        np.testing.assert_allclose(result['gap'],.5)
        np.testing.assert_allclose(result['accuracy_difference'],.5)
        np.testing.assert_allclose(result['saving_percent'],40.)

    def test_archived_predictions_match_prior_records_with_missing_confidence(self):
        sys.path.insert(0,str(Path(__file__).parent))
        from data import Evidence
        data=Evidence()
        self.assertEqual(data.checks['prior_prediction_records'],1104)
        self.assertTrue(np.isnan(data.arrays['gpt-5.6-sol']['scores']['cheap_logprob_conf']).all())


if __name__=='__main__':unittest.main()
