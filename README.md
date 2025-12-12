<p align="">
  <a href="https://frontier-cs.org">
    <img src="assets/logo.png" alt="Frontier-CS Logo" width="2000"/>
  </a>
</p>

[![Website](https://img.shields.io/badge/Website-frontier--cs.org-orange?logo=googlechrome)](https://frontier-cs.org) ![Research Problems](https://img.shields.io/badge/Research_Problems-50-blue) ![Algorithmic Problems](https://img.shields.io/badge/Algorithmic_Problems-110-green)




**Frontier-CS** is an *unsolved*, *verifiable*, *open-ended*, and *diverse* dataset for evaluating frontier models on challenging computer science problems, ranging from optimization problems in real research to competitive programming–style open challenges.

Whether you are benchmarking LLM agents, evaluating code generation models, or stress-testing reasoning capabilities, Frontier-CS provides a comprehensive suite of tasks designed for rigorous and practical evaluation.

Frontier-CS consists of two categories:

- **Algorithmic Problems**: Competitive programming challenges, including optimization, construction, and interactive problems. For each algorithmic problem, we release the full problem statement, evaluator, and *one* public test case.  

- **Research Problems**: Real-world systems challenges, including GPU kernels, distributed scheduling, ML pipelines, database optimization, and security exploits. For research problems, we release all data and scripts required to fully reproduce the results.


Frontier-CS is continuously expanding with new and increasingly challenging tasks contributed by the community.


## Quickstart

### Installation

```bash
git clone https://github.com/FrontierCS/Frontier-CS.git
cd Frontier-CS

# Install Python dependencies (using uv, recommended)
uv sync

# Or with pip:
pip install -e .
```

### API Keys

Set environment variables for the models you want to use:

```bash
export OPENAI_API_KEY="sk-..."        # For GPT models
export ANTHROPIC_API_KEY="sk-ant-..." # For Claude models
export GOOGLE_API_KEY="..."           # For Gemini models
```

## Core Components

### Research Problems

Real-world research problems requiring domain expertise in areas including machine learning, operating systems, distributed systems, GPU computing, machine learning, databases, programming languages, and security.

**Quick Start:**

```bash
cd research

# Generate solutions with an LLM
python generate_oneshot_gpt.py --model gpt-4

# Run evaluation locally (requires Docker)
./main_loop.sh

# Or run on cloud (requires SkyPilot)
python scripts/skypilot_per_solution.py --max-concurrent 4
```

### Algorithmic Problems

Competitive programming-style problems with automated judging.
**Quick Start:**

```bash
cd algorithmic

# Start judge server (requires Docker)
docker-compose up -d

# Run benchmark
python scripts/run_tests.py claude-opus-4
```

See [algorithmic/README.md](algorithmic/README.md) for details.

## Repository Structure

```
Frontier-CS/
├── research/           # Research problems
│   ├── flash_attn/
│   ├── gemm_optimization/
│   ├── cant_be_late/
│   └── ...
└── algorithmic/        # Algorithmic problems
    ├── problems/
    └── scripts/
```

## Requirements

- **Python 3.12+**
- **Docker** (for evaluation environments)
- **GPU** (optional, required for GPU-specific problems)
- **SkyPilot** (optional, for cloud-based evaluation)
- **API Keys** (for LLM solution generation)

## Contribution

We welcome contributions! To contribute new tasks:

1. **Fork the repository** and create a new branch
2. **Add your task** following the structure of existing problems
3. **Submit a Pull Request** with your task

For problems, please also send **human reference solutions** and **hidden test cases** to [qmang@berkeley.edu](mailto:qmang@berkeley.edu) to ensure evaluation integrity.

For detailed task creation guidelines, see:
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [README (Algorithmic).md](algorithmic/README.md)

## Citing Us

If you found Frontier-CS useful, please cite us as:

```bibtex

```
