# SimLayerKV

The official implementation of paper: [SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction](https://arxiv.org/pdf/2410.13846).

## Overview

Recent advancements in LLMs have extended their capabilities to handle long contexts. However, increasing the number of model layers and the length of input sequences significantly escalates the memory required to store key-value (KV) cache, posing challenges for efficient inference. To mitigate this issue, we present SimLayerKV, a simple yet effective method that reduces inter-layer KV cache redundancies by selectively dropping cache in identified lazy layers. Our approach is based on the observation that certain layers in long-context LLMs exhibit **lazy** behavior, contributing less to modeling long-range dependencies compared to non-lazy layers. By analyzing attention weight patterns, we find that the behavior of these lazy layers is consistent across tokens during generation for a given input. This insight motivates our SimLayerKV, which identifies lazy layers and reduces their KV cache accordingly. SimLayerKV is training-free, generalizable, and can be implemented with only seven lines of code. We conduct extensive experiments on three representative LLMs, e.g., LLaMA2-7B, LLaMA3-8B, and Mistral-7B across 16 tasks from the LongBench benchmark. The results demonstrate that SimLayerKV achieves a KV cache compression ratio of 5$\times$ with only a 1.2\% performance drop when combined with 4-bit quantization.

![](https://github.com/sail-sg/CPO/blob/main/Figures/crop_intro.png)

## Setup

## Reference Repositories

- LongBench [https://github.com/THUDM/LongBench](https://github.com/THUDM/LongBench)
- KIVI [https://github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)

## Citation

If you find SimLayerKV helpful or intriguing and decide to use it, kindly acknowledge the paper by citing it and consider starring this repo, thanks!

```bibtex
@misc{zhang2024simlayerkvsimpleframeworklayerlevel,
      title={SimLayerKV: A Simple Framework for Layer-Level KV Cache Reduction}, 
      author={Xuan Zhang and Cunxiao Du and Chao Du and Tianyu Pang and Wei Gao and Min Lin},
      year={2024},
      eprint={2410.13846},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13846}, 
}
