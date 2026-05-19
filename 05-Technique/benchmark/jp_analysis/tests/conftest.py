import sys
from pathlib import Path

# jp_analysis/ on sys.path so `import schema`, `import prompts...` work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
