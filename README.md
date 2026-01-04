# Interp-Toxicity

Interpretability analysis of toxicity classification models with adversarial robustness evaluation.

## Project Structure

```
interp-toxicity/
├── src/                    # Source code
│   ├── attacks/           # Adversarial attack implementations
│   │   ├── pgd_bert_attack.py
│   │   ├── pgd_hf.py
│   │   ├── bert_toxicity_attacks.py
│   │   ├── attack.py
│   │   └── adversarial.py
│   ├── models/            # Model definitions
│   │   ├── bert_classifier.py
│   │   ├── bert_classifier_vanilla.py
│   │   └── gpt2_classifier.py
│   ├── utils/             # Utility functions
│   │   └── plotly_utils.py
│   ├── patching_pytorch.py
│   ├── patching_pytorch2.py
│   ├── patching_pytorch_roberta.py
│   ├── evaluate_robustness.py
│   └── textattack_run.py
├── notebooks/             # Jupyter notebooks for analysis
│   ├── analysis.ipynb
│   ├── detection.ipynb
│   ├── generalisation.ipynb
│   ├── patching_pytorch.ipynb
│   ├── probing.ipynb
│   └── testing.ipynb
├── data/                  # Datasets
│   ├── raw/               # Raw input data
│   │   ├── jigsaw/
│   │   └── toxigen/
│   └── processed/         # Processed datasets
├── results/                # Output files
│   ├── predictions/       # Model predictions (JSON)
│   ├── scores/            # Ablation scores (PTH)
│   ├── plots/             # Visualization outputs (PNG)
│   ├── deepword/          # DeepWord attack results
│   ├── input_reduction/   # Input reduction results
│   ├── pgd/               # PGD attack results
│   ├── pruthi/            # Pruthi attack results
│   ├── pwws/              # PWWS attack results
│   ├── textfooler/        # TextFooler attack results
│   └── snlp_roberta/      # RoBERTa results
├── logs/                  # Log files and CSV outputs
├── BERT-Attack/           # BERT-Attack implementation
├── TextFooler/            # TextFooler implementation
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Poetry configuration
└── README.md             # This file
```

## Features

- **Adversarial Attacks**: Multiple attack implementations (PGD, DeepWord, Input Reduction, Pruthi, PWWS, TextFooler)
- **Model Interpretability**: Head-wise ablation analysis and patching experiments
- **Robustness Evaluation**: Comprehensive evaluation of model robustness to adversarial attacks
- **Multiple Models**: Support for BERT and GPT-2 classifiers

## Installation

```bash
# Using pip
pip install -r requirements.txt

# Using poetry
poetry install
```

## Usage

### Running Adversarial Attacks

```bash
# PGD attack
python src/attacks/pgd_hf.py --model_name bert-base-uncased --input_file data/raw/jigsaw/test.csv --output_file results/pgd_attacks.jsonl

# Using the attack runner
python src/attacks/attack.py --attack_name deepword
```

### Model Training

```bash
# Train BERT classifier
python src/models/bert_classifier.py

# Train vanilla BERT
python src/models/bert_classifier_vanilla.py
```

### Interpretability Analysis

See notebooks in `notebooks/` for:
- Head-wise ablation analysis
- Patching experiments
- Robustness evaluation
- Demographic analysis

## Key Files

- `src/attacks/pgd_bert_attack.py`: PGD-BERT attack (notebook version)
- `src/attacks/pgd_hf.py`: PGD-BERT attack (CLI version with distributed support)
- `src/patching_pytorch.py`: Patching experiments for interpretability
- `notebooks/analysis.ipynb`: Main analysis notebook

## License

[Add your license here]

## Citation

[Add citation if applicable]
