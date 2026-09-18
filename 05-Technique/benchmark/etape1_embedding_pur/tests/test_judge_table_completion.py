import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('completion', Path(__file__).parents[1] / 'scripts/124_complete_judge_tables.py')
completion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(completion)


class CompletionTest(unittest.TestCase):
    def fixture(self):
        jobs = [dict(job_id=f'{q}-{p}', qid=str(q), position=p, candidate_id_internal=str(p)) for q in range(754) for p in range(1, 11)]
        responses = [dict(job_id=j['job_id'], status='ok', judgment={'classe': 'B', 'justification': 'test'}) for j in jobs]
        return jobs, responses

    def test_fixed_k_and_duplicate_gain(self):
        jobs, responses = self.fixture()
        self.assertEqual(completion.score_responses(jobs, responses), .5)
        for j in jobs:
            j['candidate_id_internal'] = 'same'
        self.assertEqual(completion.score_responses(jobs, responses), .05)

    def test_http_error_never_zero(self):
        jobs, responses = self.fixture()
        responses[0]['status'] = 'error'
        with self.assertRaisesRegex(ValueError, 'technical error'):
            completion.score_responses(jobs, responses)

    def test_missing_or_duplicate_positions_rejected(self):
        jobs, responses = self.fixture()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            completion.score_responses(jobs, responses[:-1])
        jobs[0]['position'] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate position'):
            completion.score_responses(jobs, responses)

    def test_seed_mean_needs_three_distinct_clean_seeds(self):
        rows = [dict(retriever='LightGCN', task='article', stage='raw_retrieval', graph_label='G6', technical_status='complete_clean', seeds=s, score=.3, source_ranking_hash=str(s), ranking_identity_hash=str(s)) for s in (42, 43, 44)]
        self.assertEqual(len(completion.seed_means(rows)), 1)
        self.assertEqual(completion.seed_means(rows[:2]), [])
        rows[-1]['seeds'] = 43
        self.assertEqual(completion.seed_means(rows), [])


if __name__ == '__main__':
    unittest.main()
