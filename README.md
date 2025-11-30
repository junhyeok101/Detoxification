# Detoxification: LLM Bias and Toxicity Mitigation

## Introduction

### Problem Statement

Language models trained on biased or toxic data tend to generate harmful content, especially in role-play scenarios. The existing approaches to toxicity detection rely on simple filtering mechanisms, which fail to capture implicit biases and contextual nuances.

### Key Challenges

1. **Scarcity of Korean Toxicity Data**: Limited datasets for Korean offensive language detection
2. **Lack of Context-Aware Detoxification**: Most approaches ignore cultural context and implicit bias
3. **Implicit Bias Detection**: Difficulty in identifying and mitigating subtle forms of discrimination
4. **Role-Play Vulnerability**: LLMs are susceptible to biased prompts in conversation scenarios

### Our Hypothesis

**A model fine-tuned with SFT (Supervised Fine-Tuning) followed by DPO (Direct Preference Optimization) can suppress both explicit and implicit toxic expressions while preserving subtle tone, style, and semantic meaning.**

---

## Architecture

### System Overview

![Model_architecture](./archive/arc.png)

### 1. Data Generation Pipeline

#### Dataset Source
- **Platform**: DC Inside Gallery (Korean online community)
- **Topics**: 
  - 난민 수용 정책 (Refugee Policy)
  - 퀴어 퍼레이드 (LGBTQ+ Pride)
  - 병역제/모병제 (Military Service)
  - 여성가족부 폐지 (Gender Ministry)
  - 부동산 (Real Estate)

**Total Scale**: 500 posts per topic × 5 topics = **3,500 posts & comments**


### 2. Vector Database & RAG Construction

**Embedding Model**: `dragonkue/BGE-m3-ko` (Korean-specialized, dim=1024)
**Vector Database**: Qdrant (Fast vector search)

### 3. Model Selection & Optimization

#### Final Model: Qwen2.5-14B
**Reasons**:
- Larger capacity (14B parameters) for complex instructions
- Superior Korean language support
- Better multilingual pre-training

**Performance Comparison**:
| Benchmark | Llama-3.1-8B | Qwen2.5-14B | Gap |
|-----------|-------------|-----------|-----|
| MMLU | 69.40% | 79.70% | +10.3%p |
| MATH | 51.90% | 75.70% | +23.8%p |
| HumanEval | 72.60% | 83.50% | +10.9%p |
| IFEval | 75.0% | ~85.0% | +10%p |

### 4. Training Strategy

#### Stage 1: Supervised Fine-Tuning (SFT)

**Input**: Offensive Dataset (Crawled) + DETOX(O) pairs
**Method**: Single-turn SFT training
**Goal**: Learn basic detoxification patterns

- Supervised learning using pairs of offensive/sensitive utterances and desired detoxified responses
- Primarily uses singleton (single-turn) based standardized examples

#### Stage 2: Direct Preference Optimization (DPO)

**Input**: Offensive Dataset + DETOX(O)-1 + DETOX(O)-2 (two detoxified versions)
**Method**: Multi-turn RAG-ready DPO training
**Goal**: Learn preference between detoxified outputs while maintaining coherence

- Based on the SFT model, optimize multiple candidate responses according to preference data (human/model evaluation or automated preference)
- Learn preferences in real dialogue scenarios by including context provided by multi-turn and RAG

---

### 5. Dialogue Simulation

**Comparison Groups**:
- Base Model (Original Qwen2.5-14B)
- SFT-only Model
- SFT+DPO (2-stage) Model

**Test Methodology**:
- Set personas for each sensitive topic, input human-mimicked/crawled utterances, and simulate multi-turn dialogue
- Save conversation records (trajectory) to evaluate long-term stability

**Persona**:
- Create realistic prompts by mimicking actual community speech patterns and styles

![single](./archive/single.png)

## Results

![dpo-result](./archive/ww.png)

![Base vs Detox - Explicit Metrics](./Metrics/output/report_comparison.png)


![Implicit Bias Comparison](./Metrics/output/report_implicit_bias.png)


---

## Conclusions

### Validated Findings

1. **Biased Data Leads to Biased Outputs**: Confirmed that biased training data directly translates to toxic generation patterns
2. **DPO Provides Robustness**: SFT + DPO successfully suppresses both explicit and implicit toxic expressions
3. **Remaining Challenges**: Trade-off between response accuracy and detoxification quality exists

### Limitations

- Accuracy-Purity Trade-off: Sometimes detoxification affects semantic precision
- Persona Inconsistency: Random persona forgetting and context loss
- Static RAG Queries: Limited document retrieval diversity



---


## Quick Start

### 1. Environment Setup
```bash
conda create -n detox python=3.10 -y
conda activate detox
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install transformers peft bitsandbytes qdrant-client sentence-transformers datasets trl
```

### 2. Download Models
```bash
python3 << 'EOF'
from transformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-14B-Instruct', load_in_4bit=True)
SentenceTransformer('dragonkue/BGE-m3-ko')
EOF
```

### 3. Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant &
```

### 4. Configuration
```bash
cat > .env << 'EOF'
BASE_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct
DETOX_MODEL_NAME=./models/detox_model
EOF

mkdir -p experiment/data/personas
mkdir -p models/detox_model
```

### 5. Run Simulation
```bash
# Mode 0: Base Model
python3 experiment/run/main.py 5 0 1 A B

# Mode 1: Detox Model (SFT + DPO)
python3 experiment/run/main.py 5 1 1 A B

# Mode 2: Comparison
python3 experiment/run/main.py 5 2 1 A B
```

Arguments: `python3 experiment/run/main.py [turns] [mode] [topic] [persona1] [persona2]`
