# GlucoFM Benchmark

**Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting**

## 📊 Dataset

This benchmark uses the **GlucoFM Dataset** available on HuggingFace:

🔗 **[byluuu/gluco-tsfm-benchmark](https://huggingface.co/datasets/byluuu/gluco-tsfm-benchmark)**

The dataset includes continuous glucose monitoring (CGM) data from multiple public datasets with 80/20 train/test split. Each sample contains:
- `dataset`: Source dataset name
- `subject_id`: Subject identifier
- `timestamp`: Unix timestamp array
- `BGvalue`: Blood glucose values (mg/dL)

## 🎯 Overview

This repository provides a comprehensive benchmark for evaluating time-series foundation models on glucose forecasting tasks. It includes implementations of multiple state-of-the-art models with zero-shot, few-shot, and full-shot evaluation protocols.

**Key Features:**
- 🔄 **Multiple Training Paradigms**: Zero-shot, few-shot, and full-shot evaluation
- 📈 **Multi-Horizon Prediction**: 15min, 30min, 60min, 90min forecasting
- 🎨 **Diverse Model Architectures**: Transformer-based, LLM-based, and specialized time-series models
- 📦 **HuggingFace Integration**: Easy dataset loading and sharing
- 🔧 **Reproducible Experiments**: Documented configurations and training scripts

## 📁 Repository Structure

```
GlucoseML_benchmark/
├── 2019Martinsson_et_al_LSTM/     # LSTM baseline (Martinsson et al., 2019)
│   ├── fullshot_lstm.py           # Full-shot training & evaluation
│   ├── fewshot_lstm.py            # Few-shot training & evaluation
│   ├── datasets_loader/           # Dataset loading utilities
│   └── README.md                  # Detailed documentation
│
├── chronos-forecasting/           # Chronos-2 (Amazon)
│   ├── zeroshot.py                # Zero-shot evaluation
│   ├── fewshot.py                 # Few-shot LoRA fine-tuning
│   ├── fullshot.py                # Full-shot LoRA fine-tuning
│   └── chronos.md                 # Implementation guide
│
├── CALF/                          # CALF (Context-Aware Language Foundation)
│   ├── run.py                     # Training & evaluation script
│   ├── prepare_dataset.py         # Dataset preparation
│   ├── pca.py                     # PCA embedding generation
│   └── calf.md                    # Implementation guide
│
├── Time-LLM/                      # Time-LLM (GPT2/LLaMA-based)
│   ├── run_main.py                # Training & evaluation script
│   ├── prepare_dataset.py         # Dataset preparation
│   └── timellm.md                 # Implementation guide
│
├── GPFormer/                      # GPFormer (Graph-based Transformer)
│   ├── predict_glucose_multiwindow_gpformer_fullshot.py
│   ├── predict_glucose_multiwindow_gpformer_fewshot.py
│   └── gpformer.md                # Implementation guide
│
├── timer-model/                   # Timer (Time Series Transformer)
│   ├── predict_glucose_multiwindow_timer_zeroshot.py
│   ├── predict_glucose_multiwindow_timer_fullshot.py
│   ├── predict_glucose_multiwindow_timer_fewshot.py
│   └── timer.md                   # Implementation guide
│
├── timesfm/                       # TimesFM (Google)
│   ├── predict_glucose_multiwindow_timesfm_zeroshot.py
│   ├── predict_glucose_multiwindow_timesfm_fullshot.py
│   ├── predict_glucose_multiwindow_timesfm_fewshot.py
│   └── timesfm.md                 # Implementation guide
│
├── uni2ts/                        # Uni2TS (Moirai)
│   ├── predict_glucose_multiwindow_uni2ts_zeroshot.py
│   ├── predict_glucose_multiwindow_uni2ts_fullshot.py
│   ├── predict_glucose_multiwindow_uni2ts_fewshot.py
│   └── moirai.md                  # Implementation guide
│
└── paper_tables_ctx12h_hor30m/    # Benchmark results
    ├── zeroshot_ctx12h_hor30m_rmse.csv
    ├── fewshot_ctx12h_hor30m_rmse.csv
    └── fullshot_ctx12h_hor30m_rmse.csv
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GlucoseML_benchmark.git
cd GlucoseML_benchmark
```

### 2. Install Dependencies

Each model has its own requirements. Navigate to the specific model directory and install dependencies:

```bash
cd <model_directory>
pip install -r requirements.txt
```

### 3. Prepare Dataset

Most models support automatic HuggingFace dataset loading. For models requiring CSV format:

```bash
cd <model_directory>
python prepare_dataset.py --create_mixed
```

### 4. Run Experiments

See individual model documentation for specific commands.

## 📚 Model Documentation

### Traditional Baselines

| Model | Type | Documentation | Key Features |
|-------|------|---------------|--------------|
| **Martinsson LSTM** | LSTM | [README.md](2019Martinsson_et_al_LSTM/README.md) | Variance estimation, NLL loss, OhioT1DM baseline |

### Foundation Models (LLM-based)

| Model | Base LLM | Documentation | Key Features |
|-------|----------|---------------|--------------|
| **Time-LLM** | GPT2/LLaMA | [timellm.md](Time-LLM/timellm.md) | LLM reprogramming, time series adaptation |
| **CALF** | GPT2 | [calf.md](CALF/calf.md) | Cross-modal fine-tuning, PCA embeddings |

### Foundation Models (Transformer-based)

| Model | Architecture | Documentation | Key Features |
|-------|--------------|---------------|--------------|
| **Chronos-2** | Encoder-Decoder | [chronos.md](chronos-forecasting/chronos.md) | LoRA fine-tuning, Amazon pretrained |
| **GPFormer** | Graph Transformer | [gpformer.md](GPFormer/gpformer.md) | Multi-window prediction |
| **Timer** | Transformer | [timer.md](timer-model/timer.md) | Efficient time series modeling |
| **TimesFM** | Transformer | [timesfm.md](timesfm/timesfm.md) | Google pretrained |
| **Uni2TS (Moirai)** | Unified | [moirai.md](uni2ts/moirai.md) | Universal time series model |

## 🔬 Evaluation Protocols

### Zero-Shot Evaluation
Evaluate pretrained models without any training on glucose data.

**Supported Models:** Chronos, Timer, TimesFM, Uni2TS

**Example:**
```bash
cd chronos-forecasting
python zeroshot.py --split test --prediction_length 18
```

### Few-Shot Evaluation
Train with limited data (e.g., 1 sample per 20 hours).

**Supported Models:** All models

**Example:**
```bash
cd chronos-forecasting
python fewshot.py --train_stride 240 --prediction_length 18
```

### Full-Shot Evaluation
Train with full training dataset.

**Supported Models:** All models

**Example:**
```bash
cd chronos-forecasting
python fullshot.py --train_stride 12 --prediction_length 18
```

## 📊 Prediction Horizons

All models support multiple prediction horizons:

| Horizon | Timesteps | Duration | Use Case |
|---------|-----------|----------|----------|
| 15 min  | 3 steps   | 3 × 5min | Immediate alerts |
| 30 min  | 6 steps   | 6 × 5min | Short-term planning |
| 60 min  | 12 steps  | 12 × 5min | Meal/exercise planning |
| 90 min  | 18 steps  | 18 × 5min | Extended prediction |

**Note:** All models use 5-minute sampling frequency (standard for CGM devices).

## 📈 Benchmark Results

Results for context length = 12 hours, prediction horizon = 30 minutes are available in:

```
paper_tables_ctx12h_hor30m/
├── zeroshot_ctx12h_hor30m_rmse.csv      # Zero-shot RMSE results
├── fewshot_ctx12h_hor30m_rmse.csv       # Few-shot RMSE results
└── fullshot_ctx12h_hor30m_rmse.csv      # Full-shot RMSE results
```

## 🛠️ Common Configuration

### Context Length
Most models use **144 timesteps** (12 hours) as default context:
- 144 × 5min = 720 minutes = 12 hours

### Prediction Lengths
- `pred_len=3`: 15 minutes
- `pred_len=6`: 30 minutes
- `pred_len=12`: 60 minutes
- `pred_len=18`: 90 minutes

### Training Strategies
- **Full-shot stride**: 12 steps (1 hour) - dense sampling
- **Few-shot stride**: 240 steps (20 hours) - sparse sampling
- **Evaluation stride**: 1-3 steps (5-15 minutes) - dense prediction

## 💡 Usage Tips

1. **Start with Zero-Shot**: Test pretrained models before fine-tuning
2. **Memory Management**: Reduce batch size or use gradient accumulation for OOM errors
3. **Multi-Horizon Training**: Train once at longest horizon, evaluate at all horizons
4. **Dataset-Specific Testing**: Use test_root_path to evaluate on specific datasets
5. **HuggingFace Integration**: Most models support automatic dataset loading

## 🤝 Contributing

Contributions are welcome! To add a new model:

1. Create a new directory with model name
2. Add implementation scripts (zeroshot/fewshot/fullshot)
3. Create a `<model>.md` documentation file
4. Update this README with model information
5. Add results to `paper_tables_ctx12h_hor30m/`

## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{glucofm_benchmark2024,
  title={GlucoFM-Bench: Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting},
  author={Lu, Baiying and Liang, Zhaohui and Pontius, Ryan and Tang, Shengpu and Prioleau, Temiloluwa},
  journal={Under submission},
  year={2026}
}
```

### Individual Model Citations

**Martinsson LSTM:**
```bibtex
@article{martinsson2020blood,
  title={Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks},
  author={Martinsson, John and Schliep, Alexander and Eliasson, Bj{\"o}rn and Meijner, Claes and Persson, Simon and Mogren, Olof},
  journal={Journal of Healthcare Informatics Research},
  volume={4},
  pages={1--18},
  year={2020}
}
```

**Time-LLM:**
```bibtex
@article{jin2023time,
  title={Time-LLM: Time Series Forecasting by Reprogramming Large Language Models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and others},
  journal={arXiv preprint arXiv:2310.01728},
  year={2023}
}
```

**CALF:**
```bibtex
@article{liu2024calf,
  title={CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning},
  author={Liu, Peiyuan and Zhao, Hang and Li, Tao and others},
  journal={arXiv preprint arXiv:2403.07300},
  year={2024}
}
```

**Chronos:**
```bibtex
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Arango, Sebastian Pineda and Kapoor, Shubham and others},
  journal={arXiv preprint arXiv:2403.07815},
  year={2024}
}
@misc{ansari2025chronos2,
      title={Chronos-2: From Univariate to Universal Forecasting}, 
      author={Abdul Fatir Ansari and Oleksandr Shchur and Jaris Küken and Andreas Auer and Boran Han and Pedro Mercado and Syama Sundar Rangapuram and Huibin Shen and Lorenzo Stella and Xiyuan Zhang and Mononito Goswami and Shubham Kapoor and Danielle C. Maddix and Pablo Guerron and Tony Hu and Junming Yin and Nick Erickson and Prateek Mutalik Desai and Hao Wang and Huzefa Rangwala and George Karypis and Yuyang Wang and Michael Bohlke-Schneider},
      year={2025},
      eprint={2510.15821},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15821}, 
}
```

**Timer:**
```bibtex
@inproceedings{liutimer,
  title={Timer: Generative Pre-trained Transformers Are Large Time Series Models},
  author={Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  booktitle={Forty-first International Conference on Machine Learning}
}

@article{liu2024timer,
  title={Timer-XL: Long-Context Transformers for Unified Time Series Forecasting},
  author={Liu, Yong and Qin, Guo and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng},
  journal={arXiv preprint arXiv:2410.04803},
  year={2024}
}
```
**TimesFM:**
```bibtex
@misc{das2024decoderonlyfoundationmodeltimeseries,
      title={A decoder-only foundation model for time-series forecasting}, 
      author={Abhimanyu Das and Weihao Kong and Rajat Sen and Yichen Zhou},
      year={2024},
      eprint={2310.10688},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.10688}, 
}
```
**Moirai:**
```bibtex
@misc{woo2024unifiedtraininguniversaltime,
      title={Unified Training of Universal Time Series Forecasting Transformers}, 
      author={Gerald Woo and Chenghao Liu and Akshat Kumar and Caiming Xiong and Silvio Savarese and Doyen Sahoo},
      year={2024},
      eprint={2402.02592},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.02592}, 
}
@misc{liu2026moirai20timeseries,
      title={Moirai 2.0: When Less Is More for Time Series Forecasting}, 
      author={Chenghao Liu and Taha Aksu and Juncheng Liu and Xu Liu and Hanshu Yan and Quang Pham and Silvio Savarese and Doyen Sahoo and Caiming Xiong and Junnan Li},
      year={2026},
      eprint={2511.11698},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.11698}, 
}
```

**GPFormer**
```bibtex
@article{Zhu2025,
   author = {Taiyu Zhu and Ioannis Afentakis and Kezhi Li and Ryan Armiger and Neil Hill and Nick Oliver and Pantelis Georgiou},
   doi = {10.1109/JBHI.2024.3428921},
   issn = {21682208},
   issue = {8},
   journal = {IEEE Journal of Biomedical and Health Informatics},
   keywords = {Deep learning,Transformer,diabetes,domain generalization,glucose prediction},
   pages = {5424-5437},
   pmid = {39012743},
   publisher = {Institute of Electrical and Electronics Engineers Inc.},
   title = {Multi-Horizon Glucose Prediction Across Populations With Deep Domain Generalization},
   volume = {29},
   year = {2025}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## 📄 License

This project is licensed under the MIT License - see individual model directories for specific licenses.

