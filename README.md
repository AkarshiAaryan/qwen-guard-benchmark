# Qwen SLM Safety Guard Benchmark Suite (0.5B – 7B)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AkarshiAaryan/qwen-guard-benchmark/blob/main/notebooks/colab_pilot.ipynb)
[![Framework: PyTorch / HuggingFace](https://img.shields.io/badge/Framework-PyTorch%20%2F%20HuggingFace-orange.svg)](https://huggingface.co/)
[![Models: Qwen2.5](https://img.shields.io/badge/Models-Qwen2.5%20(0.5B%20to%207B)-blueviolet.svg)](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518fc9b5857)

A modular, reproducible evaluation framework benchmarking open-source Small Language Models (**Qwen 2.5: 0.5B, 1.5B, 3B, and 7B-Instruct**) as contextual policy safety guards in institutional, multi-tenant learning environments.

---

## 1. Executive Summary & Research Motivation

Can general-purpose open-source Small Language Models (SLMs) reliably enforce role-based access control (RBAC) and data privacy policies without domain-specific fine-tuning?

This repository benchmarks the **Qwen 2.5 model family** across a model parameter ladder from **0.5B to 7.0B**. Models are evaluated using an adversarial **Matched-Pair Methodology** on university learning management system (LMS) workflows involving multi-role institutional dynamics (Students, Teaching Assistants, Professors, System Administrators).

### Key Finding: The Failure of Scale Hypothesis
Scaling general-purpose SLMs from 0.5B to 7B **fails to reach production safety readiness**:
* **Qwen 0.5B & 1.5B** suffer from catastrophic under-refusal (failing to block 55%–65% of unauthorized actions).
* **Qwen 3B** collapses into severe hyper-refusal (falsely blocking 72.5% of legitimate requests).
* **Qwen 7B (4-bit)** achieves 72.5% pair accuracy, but still breaches 20% of safety policies, failing the strict **80.0% production deployment threshold**.

---

## 2. Decision Gate Benchmark Results

Each model was tested across **40 matched pairs (80 total scenarios)**. In a matched pair, a model must correctly **BLOCK** the violation scenario *and* correctly **ALLOW** the legitimate control scenario to score 1 point.

| Model | Parameters | Precision / Quantization | Pair Accuracy | Pairs Passed | Violations Caught (Recall) | Legitimate Allowed (Pass Rate) | Decision Gate Verdict (>=80%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Rule Baseline (Oracle)** | 0 (Rules) | None (Deterministic CPU) | **100.0%** | 40 / 40 | 40 / 40 (100.0%) | 40 / 40 (100.0%) | **DEPLOY** (Upper Bound) |
| **Qwen 2.5 0.5B** | 0.5B | FP16 | **22.5%** | 9 / 40 | 14 / 40 (35.0%) | 38 / 40 (95.0%) | **REJECT** (26 False Allows) |
| **Qwen 2.5 1.5B** | 1.5B | FP16 | **37.5%** | 15 / 40 | 18 / 40 (45.0%) | 37 / 40 (92.5%) | **REJECT** (22 False Allows) |
| **Qwen 2.5 3B** | 3.0B | FP16 | **17.5%** | 7 / 40 | 38 / 40 (95.0%) | 11 / 40 (27.5%) | **REJECT** (29 False Blocks) |
| **Qwen 2.5 7B** | 7.0B | 4-bit BitsAndBytes | **72.5%** | 29 / 40 | 32 / 40 (80.0%) | 37 / 40 (92.5%) | **REJECT** (Failed 80% Gate) |

---

## 3. Failure Mode Analysis

```
  [0.5B & 1.5B Models]               [3.0B Model]                     [7.0B Model]
  +-----------------------+          +-----------------------+        +-----------------------+
  |   Under-Refusal       |          |    Hyper-Refusal      |        |   Sub-Threshold Gate  |
  |   (False Allows)      |  ----->  |    (False Blocks)     | -----> |   8 False Allows      |
  |   Rubber-stamps       |          |    Rejects 72.5% of   |        |   3 False Blocks      |
  |   unauthorized leaks  |          |    benign workflows   |        |   72.5% Pair Accuracy |
  +-----------------------+          +-----------------------+        +-----------------------+
```

1. **Under-Refusal in Small Models (0.5B & 1.5B)**:
   Small SLMs display an inherent compliance bias. When an unauthorized user requests protected material (e.g., student accessing peer gradebook), the model defaults to helping the user, resulting in **26 False Allows** in 0.5B and **22 False Allows** in 1.5B.
2. **Hyper-Refusal Inversion in Mid-Size Models (3B)**:
   The 3B model overcorrects into indiscriminate rejection. It catches 95.0% of violations but blocks **72.5% of legitimate educational interactions** (29 False Blocks), rendering the learning assistant practically unusable.
3. **The 7B Upper Limit**:
   Qwen 7B achieves balanced reasoning (80% violation recall, 92.5% benign pass rate), but 8 violations still slip past the gate. In institutional environments, an 80% catch rate represents an unacceptable compliance vulnerability.

---

## 4. Repository Structure

```
qwen-guard-benchmark/
├── notebooks/
│   ├── colab_pilot.ipynb              # Complete, pre-computed Colab execution notebook
│   └── colab_guard_slm_100_pairs.ipynb# Full 100-pair matched evaluation harness
├── src/
│   ├── models/
│   │   ├── qwen_guard.py              # Hugging Face adapter with 4-bit quantization & greedy decoding
│   │   └── base.py                    # Abstract base classes and GuardPrediction schema
│   ├── baselines/
│   │   └── rule_baseline.py           # Ground-truth deterministic RBAC policy engine
│   ├── evaluation/
│   │   ├── pair_metrics.py            # Pairwise matched-pair scoring algorithm
│   │   ├── decision_gate.py           # Multi-model comparative decision gate generator
│   │   └── failure_analysis.py        # False Allow vs False Block taxonomy
│   ├── benchmark/
│   │   └── runner/experiment_runner.py# Batch benchmark execution runner
│   └── data/
│       └── validator.py               # Schema integrity validation
├── data/
│   ├── moodle_cases.json              # 40 matched pairs (80 total annotated cases)
│   └── GuardSLM_100_Pairs_v2.json     # Extended 100-pair evaluation dataset
├── results/
│   ├── FINAL_RESEARCH_REPORT.md       # Comprehensive research report with per-case breakdowns
│   └── slm_report.html                # Interactive, publication-ready visual report
├── tests/                             # Unit tests for adapters, schemas, and parsers
├── requirements.txt
└── README.md
```

---

## 5. Quickstart & Reproduction

### Option A: 1-Click Interactive Google Colab
Click the badge below to run the complete evaluation pipeline inside Google Colab (compatible with free T4 GPU tier):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AkarshiAaryan/qwen-guard-benchmark/blob/main/notebooks/colab_pilot.ipynb)

### Option B: Local Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/AkarshiAaryan/qwen-guard-benchmark.git
cd qwen-guard-benchmark

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest tests/
```

### Option C: Python Code Snippet (Inference with Qwen Guard Adapter)
```python
from src.models.qwen_guard import QwenGuardAdapter
from src.data.loader import TestCase

# Initialize Qwen 2.5 7B with 4-bit quantization for low-memory GPU execution
guard = QwenGuardAdapter(
    name="qwen_7b",
    config={
        "checkpoint": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True,
        "enabled": True
    }
)
guard.load()

# Predict on test case
test_case = TestCase.from_json("data/moodle_cases.json", case_id="TC01_student_own_grade")
prediction = guard.predict(test_case)

print(f"Verdict: {prediction.normalized_verdict}")  # ALLOW / BLOCK
print(f"Latency: {prediction.latency_ms} ms")
print(f"Reasoning: {prediction.raw_response}")

guard.unload()
```

---

## 6. Citation & Academic Reference

If you use this benchmark suite, datasets, or evaluation harness in your research, please cite:

```bibtex
@misc{qwen_guard_benchmark_2026,
  author = {Aaryan, Akarshi},
  title = {Qwen SLM Safety Guard Benchmark Suite: Evaluating Scale Limitations of Small Language Models in Institutional Safety Gatekeeping},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/AkarshiAaryan/qwen-guard-benchmark}}
}
```

---

## 7. License
This project is licensed under the [MIT License](LICENSE).
