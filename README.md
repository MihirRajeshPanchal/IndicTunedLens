# Indic-TunedLens: Interpreting Multilingual Models in Indian Languages

[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](https://github.com/MihirRajeshPanchal/IndicTunedLens)
[![Demo](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/MihirRajeshPanchal/IndicTunedLens)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)

A novel interpretability framework that extends the Tuned Lens approach to Indian languages by learning affine transformations for better multilingual model understanding using the Sarvam-1 model.

## 🌟 Overview

Most interpretability tools for large language models (LLMs) are designed for English, leaving a significant gap for understanding multilingual models in linguistically diverse regions like India. **Indic-TunedLens** addresses this critical limitation by:

- **Low resource Language Analysis**: Separate analysis scripts for Bengali, English, Gujarati, Hindi, Kannada, Malayalam,  Marathi, Panjabi, Nepali, Tamil and Telugu
- **Morphological Awareness**: Handling rich morphology and complex linguistic structures of Indian languages  
- **Layer wise Analysis**: Providing insights into semantic encoding across transformer layers using Sarvam-1

## 🔍 Key Findings

- **Training Languages** : Show early and stable interpretability with peak performance in layers 1-2
- **Unseen Languages** : Demonstrate delayed processing with concentrated improvements in final layers
- **Morphological Rich Languages**: Require specialized transformations for effective interpretability
- **Cross lingual Transfer**: Standard English centric methods fail to generalize to Indian languages

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/AnonymousAccountACL/IndicTunedLens.git
cd IndicTunedLens
pip install -r requirements.txt
```

### Prerequisites

1. **Sarvam-1 Model**: Download and place the Sarvam-1 model in your local directory
2. **Trained Lens**: The pre-trained lens should be in `trained_lens/sarvamai/sarvam-1/`
3. **Data**: MMLU datasets for Bengali, English, Gujarati, Hindi, Kannada, Malayalam,  Marathi, Panjabi, Nepali, Tamil and Telugu

### Training the Lens

Train the Indic-TunedLens using the provided training command:

```bash
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    -m tuned_lens train \
    sarvamai/sarvam-1 \
    /mnt/storage/deeksha/indictunedlens/sangraha.jsonl \
    --per-gpu-batch-size 1 \
    -o /mnt/storage/deeksha/indictunedlens/trained_lens/sarvamai/sarvam-1 \
    --fsdp \
    >> indictunedlens/logs/sarvamai_sarvam-1.log 2>&1
```
## 📁 Project Structure

```
  ├── README.md
  ├── download_dataset.py
  ├── gold_match.py
  ├── inference.py
  ├── merge_plots.ipynb
  ├── plot.ipynb
  ├── requirements.txt
  ├── bash/
  │   ├── 1_train.sh
  │   ├── 2_inference.sh
  │   └── 3_gold_match.sh
  └── trained_lens/
      └── sarvamai/
          └── sarvam-1/
              └── config.json
```

## 🔧 Configuration

### Model Configuration
- **Base Model**: `sarvamai/sarvam-1`
- **Hidden Size**: 2048
- **Layers**: 28
- **Vocabulary Size**: 68,096
- **Context Length**: 8,192 tokens

### Training Configuration
- **Distributed Training**: FSDP enabled
- **Batch Size**: 1 per GPU
- **Nodes**: 1
- **Processes per Node**: 5

## 📊 Analysis Output

Each analysis script generates detailed CSV files with:

- **Layer-wise predictions**: Token predictions at each transformer layer
- **Probability rankings**: Top-k token probabilities and rankings
- **Language detection**: Automatic language identification for predicted tokens
- **Position analysis**: Token position effects on interpretability


## 🗂️ Dataset Information

### Training Data [Sangraha Dataset](https://huggingface.co/datasets/ai4bharat/sangraha)

### Evaluation Data [Multilingual MMLU](https://huggingface.co/datasets/alexandrainst/m_mmlu)

<!--- 

## 📝 Citation

If you use Indic-TunedLens in your research, please cite:

```bibtex

```


--->

---

**🔗 Links**: [Paper](https://github.com/AnonymousAccountACL/IndicTunedLens) | [Demo](https://huggingface.co/spaces/AnonymousAccountACL/IndicTunedLens) | [Sarvam-1 Model](https://huggingface.co/sarvamai/sarvam-1)
