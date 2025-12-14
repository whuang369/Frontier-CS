# Research Problems: Contributing & Evaluation Guide

## Problem Structure

Each problem follows a standardized interface:

```
research/{problem}/
├── config.yaml          # Dependencies, datasets, runtime config
├── set_up_env.sh        # Environment setup
├── evaluate.sh          # Evaluation entry point
├── evaluator.py         # Scoring logic
├── readme               # Problem description
└── resources/           # Problem-specific code/data
```

### Solution Interface

Solutions implement a `Solution` class in `solution.py`:

```python
class Solution:
    def __init__(self):
        pass

    def solve(self, *args):
        # Returns: solution output (format varies by problem)
        pass
```

### Evaluation Flow

```
config.yaml → set_up_env.sh → solve.sh → evaluate.sh → evaluator.py → score (0-100)
```

---

## Adding a New Problem

### 1. Create Problem Directory

```bash
mkdir -p research/{problem_name}/resources
```

### 2. Create `config.yaml`

```yaml
tag: hpc                   # Category: os, hpc, ai, db, pl, security

dependencies:
  uv_project: resources    # Optional: uv project in resources/

datasets: []               # Optional: dataset URLs

runtime:
  timeout_seconds: 1800    # Evaluation timeout
  requires_gpu: true       # GPU requirement
  resources:               # SkyPilot resources
    accelerators: "L4:1"
    cpus: "8+"
    memory: "32+"
  environment: "CUDA 12.2, Python 3.11, PyTorch 2.0+"
```

### 3. Create Evaluation Scripts

**`set_up_env.sh`**: Prepare environment
```bash
#!/bin/bash
# Install dependencies, download data, etc.
```

**`evaluate.sh`**: Run evaluation
```bash
#!/bin/bash
python evaluator.py
```

**`evaluator.py`**: Score the solution (last line must be numeric score)
```python
# ... evaluation logic ...
print(score)  # Must be last line!
```

### 4. Register the Problem

Add to `research/problems.txt`:
```
research/{problem_name}
```

---

## Problem Hierarchy: Categories and Variants

Research problems follow a hierarchical structure:

```
Problem (e.g., gemm_optimization, poc_generation)
└── Category (e.g., squares, heap_buffer_overflow)
    └── Variant (e.g., arvo_21000)
```

| Level | Evaluation | Reporting |
|-------|------------|-----------|
| **Category** | — | Scores aggregated for leaderboard |
| **Variant** | Evaluated independently | Contributes to category score |

### Example: Simple Variants

```
research/gemm_optimization/
├── squares/           # Variant (category = squares)
│   ├── config.yaml
│   ├── readme
│   └── evaluator.py
├── rectangles/        # Variant (category = rectangles)
└── transformerish/    # Variant (category = transformerish)
```

### Example: Nested Variants

For problems with many variants per category:

```
research/poc_generation/
├── heap_buffer_overflow/       # Category
│   ├── config.yaml             # Category-level config (tag only)
│   ├── arvo_21000/             # Variant
│   │   ├── config.yaml
│   │   ├── readme
│   │   └── evaluator.py
│   └── arvo_47101/             # Variant
└── stack_buffer_overflow/      # Category
    └── ...
```

### Registering Problems

Add each **variant** (not category) to `problems.txt`:
```
research/gemm_optimization/squares
research/gemm_optimization/rectangles
research/poc_generation/heap_buffer_overflow/arvo_21000
research/poc_generation/heap_buffer_overflow/arvo_47101
```

> Note: `problems.txt` lists all evaluatable variants (109 total). The leaderboard aggregates scores by category (~50 categories).

---

## Running Evaluations

### Single Problem Evaluation

```bash
# Evaluate a single solution file
frontier-eval flash_attn solution.py

# With SkyPilot (cloud)
frontier-eval flash_attn solution.py --skypilot
```

### Batch Evaluation

```bash
# From pairs file
frontier-eval batch --pairs-file pairs.txt --results-dir results/

# From problem list × model
frontier-eval batch --model gpt-5 --problems flash_attn,cross_entropy

# With SkyPilot and parallel execution
frontier-eval batch --pairs-file pairs.txt --skypilot --max-concurrent 4

# Resume interrupted evaluation
frontier-eval batch --resume --results-dir results/
```

### Bucket Storage (Recommended for SkyPilot)

When running batch evaluations with SkyPilot, use bucket storage for reliable result persistence:

```bash
frontier-eval batch --pairs-file pairs.txt --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --max-concurrent 4
```

Each worker writes results directly to bucket as `{solution}__{problem}.json`.

### Batch Operations

#### First Run

```bash
frontier-eval batch --pairs-file pairs.txt --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --max-concurrent 4
```

#### Retry Failed Pairs

```bash
# Retry all pairs with status=error or status=timeout
frontier-eval batch --retry-failed --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --results-dir results/
```

#### Add New Problems

When `pairs.txt` or `problems.txt` is updated with new entries:

```bash
# Only evaluate pairs not yet in results
frontier-eval batch --pairs-file pairs.txt --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --resume

# Or use --complete with problems × models expansion
frontier-eval batch --complete \
    --problems-file problems.txt \
    --models-file models.txt \
    --bucket-url s3://my-bucket/frontier-results
```

#### Sync and Report Only

```bash
# Download results from bucket, generate reports, no evaluation
frontier-eval batch --sync-bucket \
    --bucket-url s3://my-bucket/frontier-results \
    --results-dir results/
```

#### Check Status

```bash
frontier-eval batch --status --results-dir results/
frontier-eval batch --report --results-dir results/
```

#### Export Failed Pairs

```bash
frontier-eval batch --export-failed failed.txt --results-dir results/
```

### Generating LLM Solutions

```bash
# Single problem
python generate_oneshot_gpt.py research/flash_attn --model gpt-5

# All problems
python generate_oneshot_gpt.py --model gpt-5

# Multiple variants
python generate_oneshot_gpt.py --model gpt-5 --variants 5
```

---

## Configuration Files

### `problems.txt`
List of problems to evaluate:
```
research/flash_attn
research/gemm_optimization/squares
research/cant_be_late/high_availability_loose_deadline
```

### `pairs.txt`
Solution-problem pairs:
```
gpt5_flash_attn:research/flash_attn
claude_gemm_squares:research/gemm_optimization/squares
```

### `config.yaml` (per problem)
Each problem has a `config.yaml` with runtime configuration:
```yaml
tag: hpc
runtime:
  docker:
    image: andylizf/triton-tlx:tlx-nv-cu122
    gpu: true
    dind: false  # Docker-in-Docker for security problems
  resources:
    accelerators: "L4:1"
    cpus: "8+"
```

### `models.txt`
Models to generate solutions for:
```
gpt-5
claude-opus-4-5
gemini-2.5-pro
```

### `num_solutions.txt`
Variant indices to generate (one per line):
```
0
1
2
```

---

## Results

### Output Files
- `results/{solution}_{problem}_result.txt`: Individual results
- `results/results.csv`: Aggregated scores
- `results/summary.txt`: Summary statistics

### Syncing Results

```bash
python scripts/results_sync.py --results-dir results
```

Rebuilds CSV, detects missing results, computes averages per model.

---

## Workflow: Typical Evaluation Cycle

```bash
# 1. Generate solutions
python generate_oneshot_gpt.py --model gpt-5

# 2. Run batch evaluation with bucket storage
frontier-eval batch --pairs-file pairs.txt --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --max-concurrent 4

# 3. Check results
frontier-eval batch --status --results-dir results/
frontier-eval batch --report --results-dir results/

# 4. Retry failed pairs
frontier-eval batch --retry-failed --skypilot \
    --bucket-url s3://my-bucket/frontier-results

# 5. Add new problems (update pairs.txt, then resume)
frontier-eval batch --pairs-file pairs.txt --skypilot \
    --bucket-url s3://my-bucket/frontier-results \
    --resume

# 6. Sync results from bucket (from any machine)
frontier-eval batch --sync-bucket \
    --bucket-url s3://my-bucket/frontier-results \
    --results-dir results/
```

---

## Result File Format

**Important**: The last line of each result file must be a numeric score.

```python
# evaluator.py
# ... evaluation logic ...
print(score)  # e.g., 85.5
```

If the last line is not numeric, the result is marked as error.

---

## CLI Reference

The `frontier-eval` CLI provides all evaluation functionality.

### Commands

```bash
# Single problem evaluation
frontier-eval <problem_id> <solution.py>
frontier-eval flash_attn solution.py --skypilot

# Batch evaluation
frontier-eval batch --pairs-file pairs.txt
frontier-eval batch --model gpt-5 --problems flash_attn,cross_entropy
frontier-eval batch --status --results-dir results/

# List problems
frontier-eval --list
frontier-eval --show flash_attn
```

### Python API

```python
from frontier_cs import FrontierCSEvaluator
from frontier_cs.batch import BatchEvaluator

# Single evaluation
evaluator = FrontierCSEvaluator()
result = evaluator.evaluate("research", "flash_attn", code=solution_code)

# Batch evaluation
batch = BatchEvaluator(results_dir="results/")
batch.evaluate_model("gpt-5", problems=["flash_attn", "cross_entropy"])
```

## Scripts Reference

Scripts in `research/`:

| Script | Description |
|--------|-------------|
| `generate_oneshot_gpt.py` | Generate solutions using LLMs |
| `scripts/results_sync.py` | Rebuild CSV from result files |
| `scripts/check_solution_matrix.py` | Verify solutions/ directory coverage |
| `scripts/submit.py` | Submit solution to evaluation server |
| `scripts/fetch.py` | Fetch evaluation result from server |

### Usage Examples

```bash
# Generate solutions for a model
python generate_oneshot_gpt.py --model gpt-5

# Check solution coverage
python scripts/check_solution_matrix.py

# Bulk submit solutions
python scripts/submit.py --submissions submissions/ --out sid_map.json

# Fetch results by sid map
python scripts/fetch.py --map sid_map.json --out results.json
```
