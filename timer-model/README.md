---
license: apache-2.0
library_name: timer
pipeline_tag: time-series-forecasting
datasets:
- thuml/UTSD
- Salesforce/lotsa_data
metrics:
- mae
- mse
tags:
- time series
- time-series
- forecasting
- foundation models
- pretrained models
- transformer
- time series foundation models
---


# Time-Series Transformer (Timer)

**Update** (2025.5): We release a generative time series foundation model [Sundial](https://arxiv.org/abs/2502.00816) on [HuggingFace](https://huggingface.co/thuml/sundial-base-128m).

**Update** (2025.2) We release [OpenLTM](https://github.com/thuml/OpenLTM) for pre-training/fine-tuning large time-series models.


**Timer** is a large time-series model introduced in this [paper](https://arxiv.org/abs/2402.02368) and enhanced by [subsequent work](https://arxiv.org/abs/2410.04803).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/nbzk91Z_yffYHmKau18qo.png)

This version is univariate pre-trained on **260B** time points with **84M** parameters, a lightweight generative Transformer for **zero-shot** point forecasting.

We evaluate the model on the following benchmark: [TSLib Dataset](https://cdn-uploads.huggingface.co/production/uploads/64fbe24a2d20ced4e91de38a/AXhZLVGR8Cnuxe8CVK4Fu.png).

For more information, please see the [GitHub](https://github.com/thuml/Large-Time-Series-Model).

There's indeed room for improvement on this model. We are actively working around it and are glad to see constructive suggestions and noteworthy cases :)

## Quickstart
```
pip install transformers==4.40.1 # Use this version and Python 3.10 for stable compatibility
```

```
import torch
from transformers import AutoModelForCausalLM

# load pretrain model
model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)

# prepare input
batch_size, lookback_length = 1, 2880
seqs = torch.randn(batch_size, lookback_length)

# generate forecast
prediction_length = 96
output = model.generate(seqs, max_new_tokens=prediction_length)

print(output.shape)
```

A notebook example is also provided [here](https://github.com/thuml/Large-Time-Series-Model/blob/main/examples/quickstart_zero_shot.ipynb). Try it out!

## Specification

* **Architecture**: Causal Transformer (Decoder-only)
* **Pre-training Scale**: 260B time points
* **Context Length**: up to 2880
* **Parameter Count**: 84M
* **Patch Length**: 96
* **Number of Layers**: 8

## Adaptation

For developers interest in fine-tune this model, we provide model checkpoint and code implementation in [OpenLTM](https://github.com/thuml/OpenLTM).

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (62022050 and U2342217), the BNRist Innovation Fund (BNR2024RC01010), and the National Engineering Research Center for Big Data Software. 

The model is mostly built from the Internet public time series dataset, which comes from different research teams and providers. We sincerely thank all individuals and organizations who have contributed the data. Without their generous sharing, this model would not have existed.


## Citation

If you find Timer or Timer-XL helpful for your research, please cite our paper:
```
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

## Contact

If you have any questions or want to use the code, feel free to contact:

* Yong Liu (liuyong21@mails.tsinghua.edu.cn)
* Guo Qin (qinguo24@mails.tsinghua.edu.cn)
 
## License

This model is licensed under the Apache-2.0 License.