import sys
from pathlib import Path

# Support both package-qualified imports at repository root and the historical
# direct-module imports used by the local operational scripts.
ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ANALYSIS_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(ANALYSIS_ROOT))
