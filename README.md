# AJNR AI Publication-Trends Pipeline

A modular pipeline for analyzing long-term trends in AI-related publications in
the **American Journal of Neuroradiology (AJNR)**, using Scopus metadata + a
**local Hugging Face LLM** (served with vLLM) and local embeddings.

It answers:

- What topics were popular / rapidly emerging each year?
- How have research themes evolved over time?
- Which papers were most cited each year, and what did they share?
- Which subfields (LLMs, multimodal, RL, agents, segmentation, reconstruction,
  radiomics, …) grew or declined?
- What new keywords / methods / benchmarks gained traction?
- Which authors and institutions had the greatest influence?

---

## 1. Install

```bash
pip install -r requirements.txt
# On the GPU host, also: pip install vllm
```

## 2. Secrets

```bash
export SCOPUS_API_KEY=...          # your key (run from an institutional IP that
                                   # subscribes to Scopus; this unlocks full
                                   # abstracts, affiliations, and references via
                                   # the Abstract Retrieval FULL view)
```
export SCOPUS_API_KEY="57581543dea4d979a748eb7941383971"

## 3. Launch the local LLM (4×48 GB recommended)

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --port 8000
```
```bash


export HF_HOME=/local/dknkim/HF
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2,3,4,5
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --port 8001


vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --port 8001

vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
   --tensor-parallel-size 4 \
   --max-model-len 32768 \
   --gpu-memory-utilization 0.90 \
   --max-num-seqs 64 \
   --reasoning-parser deepseek_r1 \
   --port 8001

export LLM_BASE_URL=http://localhost:8001/v1          # unchanged
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507     # new
export LLM_MAX_TOKENS=8192                             # NEW — essential for thinking
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
   --tensor-parallel-size 4 \
   --max-model-len 32768 \
   --gpu-memory-utilization 0.90 \
   --max-num-seqs 64 \
   --reasoning-parser deepseek_r1 \
   --port 8001





export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export LLM_MAX_TOKENS=4096

vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
   --tensor-parallel-size 4 --max-model-len 32768 \
   --gpu-memory-utilization 0.90 --max-num-seqs 64 --port 8001
```


export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export LLM_MAX_TOKENS=4096

export THINKING_LLM_BASE_URL=http://localhost:8002/v1
export THINKING_LLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
export THINKING_LLM_MAX_TOKENS=8192







Point the pipeline at it (defaults shown):

```bash
export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
export LLM_MAX_TOKENS=4096
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
    --quantization awq \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --port 8001






Step 1 — Instruct (full pipeline):
export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
export LLM_MAX_TOKENS=4096

vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
   --tensor-parallel-size 4 --max-model-len 32768 \
   --gpu-memory-utilization 0.90 --max-num-seqs 64 --port 8001
# then, in another shell:
python -c "from ajnr_ai_trends.pipeline import run_all; run_all()"
→ writes ajnr_ai_trends_report.md and ajnr_ai_trends_report_facts.json.

Step 2 — Thinking (stop Instruct, serve Thinking on the same port):
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
export LLM_MAX_TOKENS=8192

vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
   --tensor-parallel-size 4 --max-model-len 32768 \
   --gpu-memory-utilization 0.90 --max-num-seqs 64 \
   --reasoning-parser deepseek_r1 --port 8001
# then:
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507   # <-- the piece that was missing
export LLM_MAX_TOKENS=8192
python -c "from ajnr_ai_trends import report; report.report_from_facts()"



```



**Model notes.** For structured extraction, cluster labeling, and grounded
summarization, Qwen2.5-72B-Instruct is more than sufficient — the task is bound
by prompt/schema design, not raw capability, so a larger model gives little
marginal benefit. If you want stronger reasoning at the same memory budget,
`Qwen3-32B` (dense) is a clean drop-in; `Qwen3-235B-A22B-Instruct` (4-bit, TP=4)
also fits but is overkill here. Just set `LLM_MODEL` to whatever you serve.

**Embeddings.** Default `allenai/specter2_base` (citation-trained for scientific
papers; needs the `adapters` lib). Alternatives via `EMBED_MODEL`: `BAAI/bge-m3`
or `Qwen/Qwen3-Embedding-4B`.

## 4. Run

```python
from ajnr_ai_trends import pipeline
out = pipeline.run_all()          # all 10 stages, cached
```
python -c "from ajnr_ai_trends.pipeline import run_all; run_all()"

…or step through `run_pipeline.ipynb`, which runs each stage and renders the
figures inline.

---

## Pipeline stages

| # | Module | Output |
|---|--------|--------|
| 1 | `acquire`   | candidate corpus from Scopus Search API (server-side AI filter) |
| 2 | `enrich`    | full records via Abstract Retrieval API (cached) |
| 3 | `normalize` | tidy tables: papers, authors, affiliations, keywords |
| 4 | `extract`   | per-paper LLM extraction (task, methods, modality, anatomy, validation, benchmarks, reproducibility) |
| 5 | `taxonomy`  | AI subfield assignment (rules + LLM zero-shot) |
| 6 | `embed`     | local scientific-text embeddings |
| 7 | `cluster`   | unsupervised topic discovery + LLM cluster labels |
| 8 | `trends`    | per-year volume/impact, subfield growth, emergence, bursts, hype-vs-impact |
| 9 | `influence` | author / institution / country influence + co-authorship network |
| 10| `report`    | LLM-written, data-grounded markdown narrative |

All artifacts land under `data/` (`raw/`, `cache/`, `tables/`, `figures/`,
`reports/`). Caching is keyed by content, so re-runs only do new work.

---

## Recommended additional analyses (beyond the original questions)

These are implemented or scaffolded in the pipeline and worth highlighting:

1. **Hype-vs-impact quadrants** — subfield volume vs mean citations: find
   over-/under-cited areas (`trends.hype_vs_impact`).
2. **Burst detection** — keywords/methods with a concentrated surge year
   (`trends.burst_detection`).
3. **Emergence / first-appearance tracking** — when each keyword, method, and
   benchmark debuted and its trajectory (`trends.emergence`).
4. **Reproducibility & rigor trends** — share of papers reporting external
   validation, code, and public data over time (from `extract` fields). A key
   quality signal for clinical AI.
5. **Methodology shift** — CNN → transformer → foundation-model/LLM adoption
   curves (`model_family` over time).
6. **Geographic diffusion** — country participation over time
   (`influence.country_trends`).
7. **Collaboration-network evolution** — centrality of authors bridging
   subfields (`influence.coauthor_network`).
8. **Citation aging by subfield** — `citations_per_year` vs raw citations shows
   which areas age fast (e.g., early radiomics) vs sustain (e.g., LLMs).

Further ideas (not yet coded, easy extensions): co-citation/bibliographic
coupling (needs reference parsing), topic forecasting (fit a simple model to
subfield counts), and benchmarking AJNR against comparison journals (re-run
`acquire` with extra ISSNs).
```
# AJNR_AI
