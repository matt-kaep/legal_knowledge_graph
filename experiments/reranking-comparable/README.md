# E021 — reranking comparable

E021 is a prepared protocol, not an executed result. It is deliberately separate
from E016/E017 LLM-as-a-Judge: here the model changes the order of a candidate pool;
the judge only evaluates an already-produced ranking.

The frozen contract is `K_in=20` and `K_out=10` for the same 754 questions and for
three real pools: cosine/BGE-M3, PPR, and LightGCN. The same provider model, prompt,
parser, temperature, retry rule, and tie-break must be used for all three families.
No pool may be synthesized from another family. Each input pool and response must
carry a SHA-256 and the JSONL job files must be resumable by family/question.

Execution is blocked because the PPR internal-evaluation replay and a portable model
choice are not currently available in this branch. The planned minimum is 2,262
provider calls (3 × 754); token and monetary cost remains unknown until the provider
and tokenizer are frozen. No E021 score may be cited until its manifest is completed,
the inputs are hashed, and the exact Article/JP metrics pass their coverage checks.
