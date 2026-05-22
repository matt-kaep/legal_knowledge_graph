import numpy as np
import pandas as pd
from etape1.linkage import build_articles_linkage, build_jp_linkage


def test_articles_linkage_basic():
    article_ids = np.array(["code_civil:1240", "code_penal:222-23",
                            "code_penal:121-3", "cgi:1559"], dtype=object)
    article_codes = np.array(["code_civil", "code_penal", "code_penal", "cgi"], dtype=object)
    resolved_pks = {"code_penal:222-23", "code_penal:121-3"}
    penal = {"code_penal"}

    order, p2col = build_articles_linkage(article_ids, article_codes, resolved_pks, penal)
    assert order.tolist() == ["code_penal:222-23", "code_penal:121-3"]
    assert p2col.tolist() == [1, 2]
    assert (article_ids[p2col] == order).all()


def test_articles_linkage_skips_unresolved():
    article_ids = np.array(["code_penal:222-23", "code_penal:999-99"], dtype=object)
    article_codes = np.array(["code_penal", "code_penal"], dtype=object)
    resolved = {"code_penal:222-23"}
    order, p2col = build_articles_linkage(article_ids, article_codes, resolved, {"code_penal"})
    assert order.tolist() == ["code_penal:222-23"]
    assert p2col.tolist() == [0]


def test_jp_linkage_filters_no_summary():
    jp_ids = np.array(["a", "b", "c", "d"], dtype=object)
    df = pd.DataFrame({"id": ["a", "b", "c", "d"],
                       "summary": ["x", None, "y", ""]})
    order, j2row = build_jp_linkage(jp_ids, df)
    assert order.tolist() == ["a", "c"]
    assert j2row.tolist() == [0, 2]
