# Chirality-Grammar-Surgical-Anticipation

**Neuro-Symbolic Framework for Long-Horizon Surgical Action Anticipation with Probabilistic Temporal Grammar**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository implements the **Probabilistic Temporal Grammar (PTG)** framework from our HRI 2026 paper on chir ality-aware surgical action anticipation.

---

## 🎯 Overview

PTG is a neuro-symbolic framework that unifies deep video models with symbolic grammars for robust long-horizon action anticipation in surgical videos. The framework combines:

- **Neural Stage**: V-JEPA backbone + FUTR decoder for perceptual learning
- **Symbolic Stage**: Probabilistic temporal grammar with procedural, causal, and temporal rules
- **Closed Loop**: Grammar acts as regularizer during training and refiner during inference

### Key Features

- ✅ **Grammar Induction**: Hierarchical n-gram mining (~466 symbols from 117 atomic actions)
- ✅ **Chirality-Aware**: Explicit modeling of chiral action pairs (push/pull, pick/drop)
- ✅ **Multi-Dataset**: Unified meta-dataset (Cholec80, CholecT50, JIGSAWS, SAR-RARP50)
- ✅ **Robust Statistics**: Median + MAD for duration modeling
- ✅ **Goal-Conditioned**: GoMMC-based reward matrices for teleological consistency
- ✅ **Earley Parser**: Procedural validation at inference

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OFFLINE GRAMMAR INDUCTION                  │
├─────────────────────────────────────────────────────────────┤
│ • N-gram mining (2-grams, 3-grams, 4-grams)               │
│ • Hierarchical abstraction → 466 composite symbols         │
│ • Transition probabilities P (2nd-order Markov)            │
│ • Duration statistics D (median + MAD)                     │
│ • Chirality pairs γ (temporal reversals)                   │
│ • Goal matrices G, R (reachability + rewards)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   ONLINE ANTICIPATION MODEL                 │
├─────────────────────────────────────────────────────────────┤
│  V-JEPA Encoder (1024-d) → Linear Projection (512-d)       │
│                     ↓                                        │
│  FUTR Decoder (6 layers, 8 heads)                          │
│  ├─ Action Head  → P(a_t+τ)                                │
│  ├─ Duration Head → d̂                                      │
│  ├─ Object Head  → Ô (multi-label)                        │
│  └─ Goal Head    → ĝ                                       │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  NEURO-SYMBOLIC TRAINING                    │
├─────────────────────────────────────────────────────────────┤
│ Phase 1: Supervised Pre-training                           │
│   L_base = L_action + λ_dur L_duration + λ_obj L_object   │
│                                                             │
│ Phase 2: Grammar Regularization                            │
│   L_total = L_base + λ_gram KL(S_N || P_G)                │
│   where P_G = softmax(s_G / τ)                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│               INFERENCE via EARLEY PARSER                   │
├─────────────────────────────────────────────────────────────┤
│ Score(π) = Σ [log S_N(a_τ) + s_G(a_τ | h_τ)]             │
│ → Returns procedurally valid action sequences              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Chirality-Grammar-Surgical-Anticipation.git
cd Chirality-Grammar-Surgical-Anticipation

# Install dependencies
pip install -r requirements.txt
```

### Running Grammar Induction

```bash
python scripts/induce_grammar.py \
    --corpus_path data/cholec80_annotations \
    --output_path grammars/cholec80_ptg.pkl \
    --min_ngram_freq 3 \
    --max_ngram_size 4 \
    --markov_order 2
```

### Training

```bash
python scripts/train_ptg.py \
    --config configs/ptg_chirality.yaml \
    --grammar_path grammars/cholec80_ptg.pkl \
    --output_dir outputs/cholec80_run1
```

### Inference

```bash
python scripts/inference.py \
    --model_path outputs/cholec80_run1/best_model.pth \
    --grammar_path grammars/cholec80_ptg.pkl \
    --input_video path/to/video.mp4
```

---

## 📁 Project Structure

```
Chirality-Grammar-Surgical-Anticipation/
├── src/
│   ├── grammar/                 # Grammar induction
│   │   ├── ngram_miner.py      # N-gram mining + greedy matching
│   │   ├── chirality_lexicon.py # Ch iral action pairs
│   │   ├── transition_model.py  # Markov transition probabilities
│   │   ├── duration_model.py    # Robust duration statistics
│   │   └── grammar_inducer.py   # Main induction algorithm
│   │
│   ├── models/                  # Neural models
│   │   ├── vjepa_extractor.py  # V-JEPA feature extraction
│   │   ├── futr_decoder.py     # FUTR Transformer decoder
│   │   └── prediction_heads.py # Multi-task heads
│   │
│   ├── data/                    # Data pipeline
│   │   ├── cisa_dataset.py     # CiSA benchmark
│   │   └── chirality_imputation.py
│   │
│   ├── training/                # Training pipeline
│   │   ├── supervised_trainer.py
│   │   └── neuro_symbolic_trainer.py
│   │
│   └── inference/               # Inference
│       └── earley_parser.py    # Probabilistic Earley parser
│
├── configs/                     # Configurations
│   └── ptg_chirality.yaml
│
├── scripts/                     # Executable scripts
│   ├── induce_grammar.py
│   ├── train_ptg.py
│   └── inference.py
│
└── tests/                       # Unit tests
```

---

## 📊 Datasets

### CiSA Benchmark

The **Chirality in Surgical Actions (CiSA)** benchmark unifies:
- **Cholec80**: 80 cholecystectomy videos
- **CholecT50**: 50 videos with triplet annotations
- **JIGSAWS**: Robotic suturing gestures
- **SAR-RARP50**: Robotic prostatectomy

**Chiral Pairs Examples**:
- `push_needle` ↔ `pull_suture`
- `pick_tissue` ↔ `drop_tissue`
- `insert_trocar` ↔ `retract_trocar`

---

## 📈 Results

Expected performance (from paper):
- **MoC Accuracy**: ~XX% (observation α=0.2, prediction β=0.5)
- **Chirality F1**: ~XX%
- **Grammar size**: ~466 symbols (117 atomic + 349 composite)

---

## 🔬 Citation

```bibtex
@inproceedings{ferdous2026chirality,
  title={Neuro-Symbolic Anticipation with Probabilistic Temporal Grammar for Chiral Surgical Actions},
  author={Ferdous, Rezowan and others},
  booktitle={ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
  year={2026}
}
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- V-JEPA model from Facebook AI
- FUTR architecture from [Gong et al.](https://github.com/Rezowan-Ferdous/FUTR)
- KARI parser from [Gong et al.](https://github.com/gongda0e/KARI)

---

**Made with ❤️ for surgical AI research**
