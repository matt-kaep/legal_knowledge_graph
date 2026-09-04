import pytest
from budget import compute_threshold, is_oversized, BudgetError

def test_max_model_len_required():
    with pytest.raises(BudgetError):
        compute_threshold(max_model_len=None, overhead_tokens=2000, max_tokens=4000)

def test_threshold_formula():
    # 32768 - 4000 - 2000 - 512 margin = 26256
    assert compute_threshold(32768, overhead_tokens=2000, max_tokens=4000) == 26256

def test_coarse_oversized_far_above():
    thr = 26256
    assert is_oversized("x" * (3 * thr + 10_000), thr, tokenizer=None) is True

def test_coarse_clearly_below_not_oversized():
    thr = 26256
    assert is_oversized("x" * 1000, thr, tokenizer=None) is False

def test_near_threshold_uses_tokenizer():
    thr = 100
    class T:
        def encode(self, s): return list(s)   # 1 token / char
    # est = 300/3 = 100 -> dans la bande [50,150] ; tokens réels = 300 > 100
    assert is_oversized("x" * 300, thr, tokenizer=T(), band=0.5) is True
    # est = 120/3 = 40 < lo=50 -> skip cheap assumé (hors bande), même si tokens>thr
    assert is_oversized("x" * 120, thr, tokenizer=T(), band=0.5) is False
