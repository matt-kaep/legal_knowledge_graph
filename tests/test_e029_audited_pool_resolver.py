import hashlib
import importlib.util
from pathlib import Path


RESOLVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "05-Technique/benchmark/etape1_embedding_pur/scripts/e029_audited_pool_resolver.py"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location("e029_audited_pool_resolver", RESOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolves_audited_pool_from_its_declared_preflight_location(tmp_path):
    """Replacing declared audit provenance with one fixed pool root must fail this test."""
    data_root = tmp_path / "data"
    pool = data_root / "preflight_a3" / "pools" / "cosine_jp_kin50.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"qid":"q1","candidates":[]}\n', encoding="utf-8")
    audit_job_files = [
        {
            "path": str(pool),
            "sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        }
    ]

    resolver = load_resolver()

    assert resolver.resolve_audited_pool(
        audit_job_files,
        data_root=data_root,
        family="cosine",
        modality="jp",
        k_in=50,
    ) == pool
