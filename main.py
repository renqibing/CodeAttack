import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from data_preparation import DataPreparer
from judge import GPT4Judge
from post_processing import PostProcessor
from target_llm import TargetLLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ########### Target model parameters ##########
    parser.add_argument("--multi-thread", action="store_true", help="multi-thread generation")
    parser.add_argument("--max-workers", type=int, default=10, help="max workers for multi-thread generation")
    parser.add_argument("--target-model", type=str, help="Name of target model.")
    parser.add_argument("--judge-model", type=str, default="gpt-4-1106-preview", help="Name of judge model.")
    parser.add_argument("--target-max-n-tokens", type=int, default=512, help="Maximum number of generated tokens for the target.")
    parser.add_argument("--exp-name", type=str, default="main", help="Experiment file name")
    parser.add_argument("--num-samples", type=int, default=1, help="number of output samples for each prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--prompt-type", type=str, default="python_stack", help="type of adversarial prompt")
    parser.add_argument("--start-idx", type=int, default=0, help="start index of the data")
    parser.add_argument("--end-idx", type=int, default=-1, help="end index of the data")
    parser.add_argument("--query-file", type=str, default="./data/harmful_inst_cases.csv")
    parser.add_argument("--no-attack", action="store_true", help="set true when only generating adversarial examples")

    ##################################################
    args = parser.parse_args()

    # 1. Generate the prompts based on CodeAttack
    data_preparer = DataPreparer(query_file=args.query_file,
                                 prompt_name=f"code_{args.prompt_type}.txt",
                                 prompt_type=args.prompt_type)
    data_preparer.infer()

    if args.no_attack:
        # Do not run attacks
        print("Only generate adversarial examples. Don't run attacks.")
        sys.exit()

    # 2. Attack the victime model and Auto-evaluate the results
    data_key = f"code_wrapped_{args.prompt_type}"
    query_name = Path(args.query_file).stem
    with open(f"./prompts/data_{query_name}_{args.prompt_type}.json") as f:
        datas = json.load(f)
        if args.end_idx == -1:
            args.end_idx = len(datas)
        datas = datas[args.start_idx: args.end_idx]

    # TODO: do we really need to initialize this list?
    results = [{} for _ in range(len(datas))]

    judgeLLM = GPT4Judge(args.judge_model)
    postprocessor = PostProcessor(args.prompt_type)

    def func_wrap(idx, data, targetLLM=None):
        if not targetLLM:
            targetLLM = TargetLLM(
                args.target_model,
                args.target_max_n_tokens,
                temperature=args.temperature,
            )

        question = data[data_key]
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

            # evaluate by JudgeLLM
            score, reason = judgeLLM.infer(plain_attack, resp)
            results[idx]["judge_score"] = score
            results[idx]["judge_reason"] = reason

        results[idx]["qA_pairs"].append(
            {"Q": question, "A": target_response_list}
        )
        print("===========================================\n")
        print("idx", idx)
        return

    if args.multi_thread:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = executor.map(func_wrap,
                                   list(range(len(datas))),
                                   datas)
    else:
        targetLLM = TargetLLM(
            model_name=args.target_model,
            max_tokens=args.target_max_n_tokens,
            temperature=args.temperature
        )
        for idx, data in enumerate(datas):
            func_wrap(idx, data, targetLLM)

    results_dumped = json.dumps(results)
    os.makedirs("results", exist_ok=True)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_model = args.target_model.split("/")[-1]
    res_filename = (f"./results/{target_model}_{args.exp_name}_{data_key}"
                    f"_temp_{args.temperature}_results_{cur_time}.json")
    with open(res_filename, "w+", ) as f:
        f.write(results_dumped)

    print(f"Results added in file {res_filename}")
