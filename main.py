import argparse
import json
import os
import time
import openai
import anthropic
import replicate
from typing import List, Union, Dict
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from judge import GPT4Judge
from target_llm import TargetLLM
from data_preparation import DataPreparer
from post_processing import PostProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ########### Target model parameters ##########
    parser.add_argument("--multi-thread", action="store_true", help="multi-thread generation")
    parser.add_argument("--max-workers", type=int, default=10, help="max workers for multi-thread generation")
    parser.add_argument("--target-model", default="", help="Name of target model.")
    parser.add_argument("--judge-model", default="gpt-4-1106-preview", help="Name of judge model.")
    parser.add_argument("--target-max-n-tokens", type=int, default=0, help="Maximum number of generated tokens for the target.")
    parser.add_argument("--exp-name", type=str, default="main", help="Experiment file name")
    parser.add_argument("--num-samples", type=int, default=1, help="number of output samples for each prompt")
    parser.add_argument("--judge", action="store_true", help="whether to use LLMs to judge the generated response")
    parser.add_argument("--data_key", type=str, default="plain_attack", help="data key for the experiment")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--start-idx", type=int, default=0, help="start index of the data")
    parser.add_argument("--prompt-type", type=str, default="python_stack", help="type of adversarial prompt")
    parser.add_argument("--end-idx", type=int, default=-1, help="end index of the data")
    parser.add_argument("--query-file", type=str, default="./data/harmful_inst_cases.csv")
    parser.add_argument("--no-attack", action="store_true", help="set true when only generating adversarial examples")

    ##################################################
    args = parser.parse_args()

    # 1. Generate the prompts based on CodeAttack 
    data_preparer = DataPreparer(query_file=args.query_file, prompt_name=f"code_{args.prompt_type}.txt", prompt_type=args.prompt_type)
    data_preparer.infer()

    if not args.no_attack:
        # 2. Attack the victime model and Auto-evaluate the results
        args.data_key = f"code_wrapped_{args.prompt_type}"
        query_name = query_file.split('/')[-1].split('.')[0]
        f = open(
            f"./prompts/data_{query_name}_{args.prompt_type}.json",
        )

        datas = json.load(f)
        if args.end_idx == -1:
            args.end_idx = len(datas)
        datas = datas[args.start_idx : args.end_idx]
        f.close()
        results = [{} for _ in range(len(datas))]
        
        judgeLLM = GPT4Judge(args.judge_model, prompt_type=args.prompt_type)
        postprocessor = PostProcessor(args.prompt_type)

        def func_wrap(idx, data, targetLLM=None):
            if targetLLM is None:
                targetLLM = TargetLLM(
                    args.target_model,
                    args.target_max_n_tokens,
                    temperature=args.temperature,
                )

            question = data[args.data_key]
            plain_attack = data["plain_attack"]

            results[idx]["plain_attack"] = plain_attack
            results[idx]["qA_pairs"] = []

            target_response_list = []
            # random attack for args.num_samples times
            for j in range(args.num_samples):
                # attack the victim model 
                target_response = targetLLM.generate(question)
                target_response_list.append(target_response)
                # extract the harmful content from the generated code 
                resp = postprocessor.core(target_response)
                results[idx]["res_simplified"] = resp
                if args.judge:
                    # evaluate by JudgeLLM 
                    score, reason = judgeLLM.infer(plain_attack, resp)
                    results[idx]["judge_score"] = score
                    results[idx]["judge_reason"] = reason
            results[idx]["qA_pairs"].append({"Q": question, "A": target_response_list})
            print("===========================================\n")
            print("idx", idx)
            return

        if args.multi_thread:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = executor.map(func_wrap, list(range(len(datas))), datas)
        else:
            targetLLM = TargetLLM(
                args.target_model, args.target_max_n_tokens, temperature=args.temperature
            )
            for idx, data in enumerate(datas):
                func_wrap(idx, data, targetLLM)

        results_dumped = json.dumps(results)
        os.makedirs("results", exist_ok=True)
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_model = args.target_model.split("/")[-1]
        with open(
            f"./results/{target_model}_{args.exp_name}_{args.data_key}_temp_{args.temperature}_results_{cur_time}.json",
            "w+",
        ) as f:
            f.write(results_dumped)
        f.close()