<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (ICLR'24) Time-LLM: Time Series Forecasting by Reprogramming Large Language Models </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/KimMeen/Time-LLM?color=green)
![](https://img.shields.io/github/stars/KimMeen/Time-LLM?color=yellow)
![](https://img.shields.io/github/forks/KimMeen/Time-LLM?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2310.01728">Paper Page</a>]**
**[<a href="https://www.youtube.com/watch?v=6sFiNExS3nI">YouTube Talk 1</a>]**
**[<a href="https://www.youtube.com/watch?v=L-hRexVa32k">YouTube Talk 2</a>]**
**[<a href="https://medium.com/towards-data-science/time-llm-reprogram-an-llm-for-time-series-forecasting-e2558087b8ac">Medium Blog</a>]**

**[<a href="https://www.jiqizhixin.com/articles/2024-04-15?from=synced&keyword=TIME-LLM">机器之心中文解读</a>]**
**[<a href="https://mp.weixin.qq.com/s/UL_Kl0PzgfYHOnq7d3vM8Q">量子位中文解读</a>]**
**[<a href="https://mp.weixin.qq.com/s/FSxUdvPI713J2LiHnNaFCw">时序人中文解读</a>]**
**[<a href="https://mp.weixin.qq.com/s/nUiQGnHOkWznoBPqM0KHXg">AI算法厨房中文解读</a>]**
**[<a href="https://zhuanlan.zhihu.com/p/676256783">知乎中文解读</a>]**


</div>

<p align="center">

<img src="./figures/logo.png" width="70">

</p>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@inproceedings{jin2023time,
  title={{Time-LLM}: Time series forecasting by reprogramming large language models},
  author={Jin, Ming and Wang, Shiyu and Ma, Lintao and Chu, Zhixuan and Zhang, James Y and Shi, Xiaoming and Chen, Pin-Yu and Liang, Yuxuan and Li, Yuan-Fang and Pan, Shirui and Wen, Qingsong},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Updates/News:
🚩 **News** (Oct. 2025): Time-LLM has been cited 1,000 times in the past two years! 🎉 We are deeply grateful to the community for the incredible support along the journey.

🚩 **News** (Aug. 2024): Time-LLM has been adopted by XiMou Optimization Technology Co., Ltd. (XMO) for Solar, Wind, and Weather Forecasting.

🚩 **News** (Oct. 2024): Time-LLM has been included in [PyPOTS](https://pypots.com/). Many thanks to the PyPOTS team!

🚩 **News** (May 2024): Time-LLM has been included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributor @[JQGoh](https://github.com/JQGoh) and @[marcopeix](https://github.com/marcopeix)!

🚩 **News** (Mar. 2024): Time-LLM has been upgraded to serve as a general framework for repurposing a wide range of language models to time series forecasting. It now defaults to supporting Llama-7B and includes compatibility with two additional smaller PLMs (GPT-2 and BERT). Simply adjust `--llm_model` and `--llm_dim` to switch backbones.

## Introduction
Time-LLM is a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, we show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.

<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>

- Time-LLM comprises two key components: (1) reprogramming the input time series into text prototype representations that are more natural for the LLM, and (2) augmenting the input context with declarative prompts (e.g., domain expert knowledge and task instructions) to guide LLM reasoning.

<p align="center">
<img src="./figures/method-detailed-illustration.png" height = "190" alt="" align=center />
</p>

## Requirements
Use python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Tune the model. We provide five experiment scripts for demonstration purpose under the folder `./scripts`. For example, you can evaluate on ETT datasets by:

```bash
bash ./scripts/TimeLLM_ETTh1.sh 
```
```bash
bash ./scripts/TimeLLM_ETTh2.sh 
```
```bash
bash ./scripts/TimeLLM_ETTm1.sh 
```
```bash
bash ./scripts/TimeLLM_ETTm2.sh
```

## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.


## Further Reading

As one of the earliest works exploring the intersection of large language models and time series, we sincerely thank the open-source community for supporting our research. While we do not plan to make major updates to the main Time-LLM codebase, we still welcome **constructive pull requests** to help maintain and improve it.

🌟 Please check out our team’s latest research projects listed below. 

1, [**TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models**](https://arxiv.org/pdf/2509.24803), *arXiv* 2025.

**Authors**: Tong Guan, Zijie Meng, Dianqi Li, Shiyu Wang, Chao-Han Huck Yang, Qingsong Wen, Zuozhu Liu, Sabato Marco Siniscalchi, Ming Jin, Shirui Pan

```bibtex
@article{guan2025timeomni,
  title={TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models},
  author={Guan, Tong and Meng, Zijie and Li, Dianqi and Wang, Shiyu and Yang, Chao-Han Huck and Wen, Qingsong and Liu, Zuozhu and Siniscalchi, Sabato Marco and Jin, Ming and Pan, Shirui},
  journal={arXiv preprint arXiv:2509.24803},
  year={2025}
}
```

2, [**Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement**](https://arxiv.org/pdf/2503.01875), in *ACL* 2025.
[\[HuggingFace\]](https://huggingface.co/Time-MQA)

**Authors**: Yaxuan Kong, Yiyuan Yang, Yoontae Hwang, Wenjie Du, Stefan Zohren, Zhangyang Wang, Ming Jin, Qingsong Wen

```bibtex
@inproceedings{kong2025time,
  title={Time-mqa: Time series multi-task question answering with context enhancement},
  author={Kong, Yaxuan and Yang, Yiyuan and Hwang, Yoontae and Du, Wenjie and Zohren, Stefan and Wang, Zhangyang and Jin, Ming and Wen, Qingsong},
  booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year={2025}
}
```

3, [**Towards Neural Scaling Laws for Time Series Foundation Models**](https://arxiv.org/pdf/2410.12360), in *ICLR* 2025.
[\[GitHub Repo\]](https://github.com/Qingrenn/TSFM-ScalingLaws)

**Authors**: Qingren Yao, Chao-Han Huck Yang, Renhe Jiang, Yuxuan Liang, Ming Jin, Shirui Pan

```bibtex
@inproceedings{yaotowards,
  title={Towards Neural Scaling Laws for Time Series Foundation Models},
  author={Yao, Qingren and Yang, Chao-Han Huck and Jiang, Renhe and Liang, Yuxuan and Jin, Ming and Pan, Shirui},
  booktitle={International Conference on Learning Representations (ICLR)}
  year={2025}
}
```

4, [**Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**](https://arxiv.org/pdf/2409.16040), in *ICLR* 2025.
[\[GitHub Repo\]](https://github.com/Time-MoE/Time-MoE)

**Authors**: Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, Ming Jin

```bibtex
@inproceedings{shi2024time,
  title={Time-moe: Billion-scale time series foundation models with mixture of experts},
  author={Shi, Xiaoming and Wang, Shiyu and Nie, Yuqi and Li, Dianqi and Ye, Zhou and Wen, Qingsong and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

5, [**TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis**](https://arxiv.org/abs/2410.16032), in *ICLR* 2025.
[\[GitHub Repo\]](https://github.com/kwuking/TimeMixer/blob/main/README.md)

**Authors**: Shiyu Wang, Jiawei Li, Xiaoming Shi, Zhou Ye, Baichuan Mo, Wenze Lin, Shengtong Ju, Zhixuan Chu, Ming Jin

```bibtex
@inproceedings{wang2024timemixer++,
  title={TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis},
  author={Wang, Shiyu and Li, Jiawei and Shi, Xiaoming and Ye, Zhou and Mo, Baichuan and Lin, Wenze and Ju, Shengtong and Chu, Zhixuan and Jin, Ming},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [OFA (GPT4TS)](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and have extensively modified it to our purposes. We thank the authors for sharing their implementations and related resources.
