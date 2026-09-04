import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "32_lightgcn_strict.py"
)
spec = importlib.util.spec_from_file_location("lightgcn_strict_for_tests", SCRIPT)
lightgcn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lightgcn)


def test_hard_negative_cosine_topn_removes_all_gold_items():
    q_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    item_emb = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    item_indices = np.array([100, 101, 102, 103], dtype=np.int64)
    gold_by_question = {0: {100, 101}}

    pools = lightgcn.build_hard_negative_pools(
        q_emb,
        item_emb,
        item_indices,
        gold_by_question,
        top_n=3,
    )

    assert pools[0].tolist() == [102]


def test_sample_negative_items_uses_hard_pool_then_random_fallback():
    pos_q = np.array([0, 1], dtype=np.int64)
    item_indices = np.array([100, 101, 102, 103], dtype=np.int64)
    gold_by_question = {0: {100}, 1: {103}}
    hard_pools = {0: np.array([102], dtype=np.int64), 1: np.array([], dtype=np.int64)}
    rng = np.random.default_rng(123)

    sampled = lightgcn.sample_negative_items(
        pos_q,
        item_indices,
        gold_by_question,
        rng,
        hard_negative_pools=hard_pools,
    )

    assert sampled[0] == 102
    assert sampled[1] in {100, 101, 102}
    assert sampled[1] != 103


def test_parse_negative_sampling_strategy_accepts_semihard_rank_window():
    assert lightgcn.parse_negative_sampling_strategy(
        "semi_hard_cosine_rank21_50"
    ) == ("semi_hard_cosine_rank", (21, 50))


def test_semi_hard_pool_uses_original_cosine_ranks_21_to_50_and_removes_gold():
    q_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    item_emb = np.array(
        [[1.0, rank / 100.0] for rank in range(1, 56)],
        dtype=np.float32,
    )
    item_indices = np.arange(1, 56, dtype=np.int64)

    pools = lightgcn.build_cosine_negative_pools(
        q_emb,
        item_emb,
        item_indices,
        {0: {21, 24}},
        start_rank=21,
        end_rank=50,
    )

    expected = [22, 23, *range(25, 51)]
    assert pools[0].tolist() == expected
    assert 21 not in pools[0]
    assert 24 not in pools[0]
    assert not set(range(1, 21)) & set(pools[0].tolist())


def test_similarity_batches_match_one_shot_product_without_full_score_matrix():
    queries = lightgcn.torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=lightgcn.torch.float32
    )
    candidates = lightgcn.torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=lightgcn.torch.float32
    )

    batches = list(lightgcn.iter_similarity_batches(queries, candidates, batch_size=2))

    assert [start for start, _ in batches] == [0, 2]
    np.testing.assert_allclose(
        np.vstack([scores for _, scores in batches]),
        (queries @ candidates.T).numpy(),
    )
