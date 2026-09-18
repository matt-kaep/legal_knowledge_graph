from etape1.tokenize_stats import compute_token_stats


class FakeTokenizer:
    """Tokenizer factice : 1 token par caractère."""

    def __call__(self, texts, **kw):
        return {"input_ids": [list(range(len(t))) for t in texts]}


def test_basic_stats():
    texts = ["a", "ab", "abc", "abcd"]  # longueurs 1, 2, 3, 4
    s = compute_token_stats(texts, FakeTokenizer(), max_ctx=3)
    assert s["n"] == 4
    assert s["p50"] in (2, 3)
    assert s["p100"] == 4
    assert s["n_over_ctx"] == 1  # seul "abcd" dépasse 3
    assert s["pct_over_ctx"] == 0.25
