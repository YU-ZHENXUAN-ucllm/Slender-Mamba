## Project Title

This repository contains the code and supplementary materials for the following paper:
> Zhenxuan Yu, Takeshi Kojima, Yutaka Matsuo and Yusuke Iwasawa" Slender-Mamba: Fully Quantized  Mamba in 1.58 Bits From Head to Toe," The 31st International Conference on Computational Linguistics (COLING 2025).


## Installation
Installation instructions are heavily inspired by the original [state-spaces/mamba](https://github.com/state-spaces/mamba) repo.

- [Optional] `pip install causal_conv1d==1.1.1`: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.
- `pip install mamba-ssm`: the core Mamba package.
- `pip install flash-attn==2.5.8`: only used for the hybrid Mamba model


## Examples

To test generation latency (e.g. batch size = 1) with different sampling strategies:

``` sh
python generation.py --model-name "slender-mamba" --prompt "Whenever I try to understand quantum mechanics" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
```

## Pre-trained Models 

released soon.

## Evaluations 

released soon.

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{Yu2025Slender-Mamba,
  title={Slender-Mamba: Fully Quantized  Mamba in 1.58 Bits From Head to Toe},
  author={Zhenxuan Yu, Takeshi Kojima, Yutaka Matsuo and Yusuke Iwasawa},
  booktitle={The 31st International Conference on Computational Linguistics (COLING 2025)},
  year={2025}
}