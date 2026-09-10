from etape1.eval_recall import (
    recall_at_k, kstar, extract_pourvoi_numbers,
)


def test_recall_at_k_basic():
    ranked = ["a", "b", "c", "d", "e"]
    gold = {"b", "e"}
    assert recall_at_k(ranked, gold, k=1) == 0.0
    assert recall_at_k(ranked, gold, k=2) == 0.5
    assert recall_at_k(ranked, gold, k=5) == 1.0


def test_recall_at_k_no_gold():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_kstar_threshold():
    ranked = ["x", "y", "a"]
    gold = {"a", "b"}
    assert kstar(ranked, gold, ks=[1, 2, 3, 5], threshold=0.5) == 3


def test_kstar_never_reached():
    ranked = ["x", "y"]
    gold = {"a"}
    assert kstar(ranked, gold, ks=[1, 2], threshold=0.5) is None


def test_extract_pourvoi_cc_format():
    text = "Cass. crim., 9 janv. 2019, n 18-82.829"
    assert extract_pourvoi_numbers(text) == ["18-82.829"]


def test_extract_pourvoi_multiple_formats():
    text = "Cass. crim., n° 20-80.135 et n. 90-83.786"
    nums = extract_pourvoi_numbers(text)
    assert "20-80.135" in nums
    assert "90-83.786" in nums
