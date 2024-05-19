<h1 align="center"> CodeAttack 🧑‍💻🐞</h1>
A novel framework CodeAttack to systematically investigate the safety vulnerability issues of LLMs in the domain of code. 
<br>   <br>
<h3 align="center">RESEARCH USE ONLY✅ NO MISUSE❌</h3>
<h3 align="center">LOVE💗 and Peace🌊</h3>

## 🆙 Updates
- An enhanced version of CodeAttack, highly effective against the latest `GPT-4` and `Claude-3` series models, will be available next week!

## 👉 Paper
For more details, please refer to our paper [ACL 2024](https://arxiv.org/abs/2403.07865).

## 🛠️ Usage
✨An example run:
```
python3 main.py --num_sample 3 \
--prompt-type code_python \ 
--target-model=gpt-4-1106-preview \
--judge-model=gpt-4-1106-preview \
--exp_name=python_stack_full \
--data_key=code_wrapped_plain_attack \
--target-max-n-tokens=1000 \
--judge \
--multi-thread \
--temperature 0 \
--start_idx 0 --end_idx -1
```
### Experiments 
1. The 'data' folder contains the CodeAttack datasets curated using [AdvBench](https://github.com/llm-attacks/llm-attacks). There are three versions of CodeAttack, each utilizing different input encoding ways: `data_python_string_full.json`, `data_python_list_full.json`, and `data_python_stack_full.json`.
2. We provide templates for our CodeAttack in both C++ and Go, named `code_C_string.txt` and `code_go_string.txt`, respectively.
3. Given the limited capability of Llama-2-7B-chat to follow instructions within the code domain, we simplify our CodeAttack in the form of `code_python_list_llama.txt` for evaluation purposes. We also present manual evaluation results of Llama-2-7B-chat in our paper, instead of using GPT-4 as the evaluator.


## 💡Framework
<div align="center">
  <img src="figs/main.png" alt="Logo" width="500">
</div>


Overview of our CodeAttack. CodeAttack constructs a code template with three steps: (1) Input encoding which encodes the harmful text-based query with common data structures; (2) Task understanding which applies a decode() function to allow LLMs to extract the target task from various kinds of inputs; (3) Output specification which enables LLMs to fill the output structure with the user’s desired content.

## Citation

If you find our paper&tool interesting and useful, please feel free to give us a star and cite us through:
```bibtex
@inproceedings{
Ren2024codeattack,
title={Exploring Safety Generalization Challenges of Large Language Models via Code},
author={Qibing Ren and Chang Gao and Jing Shao and Junchi Yan and Xin Tan and Wai Lam and Lizhuang Ma},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
url={https://arxiv.org/abs/2403.07865}
}

```