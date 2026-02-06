# GlucoFM-Bench – Chronos Evaluation

Run zero-shot, few-shot, and full-shot Chronos-2 on the HuggingFace CGM benchmark dataset.

Dataset used:
byluuu/gluco-tsfm-benchmark

---

## 1. Setup

Create environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages:
```bash
pip install datasets
pip install chronos-forecasting
pip install matplotlib
```

## 2. Zero-shot
```bash
python zeroshot.py \
  --split test \
  --step_size 1 \
  --prediction_length 18
```
## 3. Few-shot (LoRA fine-tuning)
```bash
python fewshot.py \
  --train_split train \
  --test_split test \
  --context_length 144 \
  --prediction_length 18 \
  --train_stride 240 \
  --batch_size 32 \
  --learning_rate 1e-5
```
## 4. Full-shot (LoRA fine-tuning)
```bash
python fullshot.py \
  --train_split train \
  --test_split test \
  --context_length 144 \
  --prediction_length 18 \
  --train_stride 12 \
  --batch_size 32 \
  --learning_rate 1e-5
```

## 5. Output
Saved to:
```css
results/<dataset_name>/
  summary.csv
  patient_results/

```

