"""
This script performs evaluation of a language model on various datasets. It includes functions to generate prompts, run inference, and evaluate the model's performance.
Functions:
    get_prompt_list(args):
        Generates a list of prompts based on the evaluation dataset specified in the arguments.
        Args:
            args: The arguments containing evaluation dataset and model details.
        Returns:
            prompt_list: A list of prompts generated from the input data.
            input_datapath: The path to the input data file.
    main():
        The main function that orchestrates the evaluation process.
        It loads the model, generates prompts, runs inference, and evaluates the results.
        It also saves the outputs, prompts, and evaluation metrics to JSON files.
"""

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams
from arguments import get_args
from dataset import load_data, get_inputs
import torch
from datasets import load_dataset
import os
from get_scores import *
from evaluation_utils import *
import numpy as np
import json


def get_prompt_list(args):

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # get input data
    input_datapath = None  # 初始化 input_datapath
    if args.eval_dataset == "nqopen":
        input_datapath = os.path.join(
            args.data_folder, "nq_open/nq-open-oracle.json")
    elif args.eval_dataset == "nqswap":
        pass
    elif args.eval_dataset == "eli5":
        input_datapath = os.path.join(
            args.data_folder,
            "eli5/eli5_eval_bm25_top100_reranked_oracle.json")
    elif args.eval_dataset == "xsum":
        input_datapath = os.path.join(args.data_folder, "xsum/xsum-1000.jsonl")
    elif args.eval_dataset == "quac":
        input_datapath = os.path.join(args.data_folder, "quac/test.json")
    elif args.eval_dataset == "triviaqa":
        input_datapath = os.path.join(
            args.data_folder, "triviaqa/web-dev-top200.json")
    elif args.eval_dataset == "memo-trap":
        input_datapath = os.path.join(args.data_folder, "memo_trap")
    else:
        raise Exception("please input a correct eval_dataset name!")

    if args.eval_dataset == "nqswap":
        ds = load_dataset("pminervini/NQ-Swap")
        data_list = ds['dev']
        prompt_list = []
        for data in data_list:
            # 提取source_info中的question和passages
            question = data['question']
            passage = data['sub_context']
            question = f"Based on the following text:\n{passage}\nAnswer the following question: {question}\nPlease directly response the accurate object"
            question = [
                {"role": "system", "content": ""},
                {"role": "user", "content": question, }
            ]
            prompt_list.append(
                tokenizer.apply_chat_template(
                    question,
                    tokenize=False,
                    add_generation_prompt=True))
    elif args.eval_dataset == "xsum":
        prompt_list = []
        with open(input_datapath, 'r') as f:
            data_list = [json.loads(line) for line in f]
            instruction = "Generate a summary comprising of 1 sentence for the given article.\n\n"
            for data in data_list:
                prompt_list.append(instruction +
                                   f"Article: {data['document']}\n\nSummary:")
    elif args.eval_dataset == "memo-trap":
        prompt_list = []
        for file_name in [
            "1-proverb-ending.csv",
            "2-proverb-translation.csv",
            "3-hate-speech-ending.csv",
                "4-history-of-science-qa.csv"]:
            # for file_name in ["4-history-of-science-qa.csv"]:
            file_path = os.path.join(input_datapath, file_name)
            data = pd.read_csv(file_path)
            data_list = data.iloc[:, 0]  # Extract the first column
            for prompt in data_list:
                question = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt, }
                ]
                prompt_list.append(
                    tokenizer.apply_chat_template(
                        question,
                        tokenize=False,
                        add_generation_prompt=True))
    elif args.eval_dataset == "eli5":
        prompt_list = []
        data_list = load_data(input_datapath)
        print("number of samples in the dataset:", len(data_list))
        for data in data_list:
            question = [
                {
                    "role": "system",
                    "content": ""},
                {
                    "role": "user",
                    "content": "Base on the following claims:\n1." +
                    data["claims"][0] +
                    "\n2." +
                    data["claims"][1] +
                    "\n3." +
                    data["claims"][2] +
                    "\nAnswer the following question: " +
                    data["question"]},
            ]
            prompt_list.append(
                tokenizer.apply_chat_template(
                    question,
                    tokenize=False,
                    add_generation_prompt=True))
    else:
        data_list = load_data(input_datapath)
        print("number of samples in the dataset:", len(data_list))
        prompt_list = get_inputs(
            data_list,
            args.eval_dataset,
            tokenizer,
            num_ctx=args.num_ctx,
            max_output_len=args.out_seq_len)

    return prompt_list, input_datapath


def main():
    args = get_args()

    # get model_path
    model_path = args.model_id

    # get prompt_list
    prompt_list, ground_truth_file = get_prompt_list(args)

    # get output_datapath
    output_datapath = os.path.join(
        args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_output.txt")

    # run inference
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=1,
        max_tokens=args.max_tokens)

    # This changes the GPU support to 8WSW
    model_vllm = LLM(
        model_path,
        tensor_parallel_size=args.gpu_num,
        gpu_memory_utilization=0.4)

    all_metrics = []

    for i in range(1):  # set to 3 for 3 times evaluation
        outputs = model_vllm.generate(prompt_list, sampling_params)
        output_list = []
        for output in outputs:
            output_list.append(
                output.outputs[0].text.strip().replace(
                    "\n", " "))

        # Write output and prompt lists to a JSON file
        output_json_path = os.path.join(
            args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_output_prompt_{i + 1}.json")
        result = []
        ground_truths = get_sub_answers(args.eval_dataset, ground_truth_file)
        if args.eval_dataset == "xsum":
            ground_truths = ["No answer"] * len(prompt_list)
        for prompt, output, ground_truth in zip(
                prompt_list, output_list, ground_truths):
            result.append({"prompt": prompt, "output": output,
                          "ground_truth": ground_truth})
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        # Save args to a JSON file
        args_json_path = os.path.join(
            args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_args_{i + 1}.json")
        with open(args_json_path, 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)

        metric_save_file_path = os.path.join(
            args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_metrics_{i + 1}.json")

        prediction_file = output_datapath
        print("-" * 80)
        if args.eval_dataset == "quac":
            precision, recall, f1 = evaluate_f1(
                ground_truth_file, prediction_file)
            metric = {"precision": precision, "recall": recall, "f1": f1}
        elif args.eval_dataset == "nqopen" or args.eval_dataset == "triviaqa" or args.eval_dataset == "memo-trap":
            text_column = output_list
            sub_answers = get_sub_answers(args.eval_dataset, ground_truth_file)

            assert len(text_column) == len(
                sub_answers), "The lengths of the columns do not match."

            data = []
            for text, sub_answer in zip(text_column, sub_answers):
                data.append({
                    'predicted_answer': text,
                    'org_answer': [''],
                    'sub_answer': sub_answer
                })
            metric = em_and_subem(data, args.eval_dataset)
            zero_subspan_em_indices = get_zero_subspan_em_indices(
                data, args.eval_dataset)
            zero_subspan_em_results = [result[idx]
                                       for idx in zero_subspan_em_indices]
            zero_subspan_em_json_path = os.path.join(
                args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_zero_subspan_em.json")
            with open(zero_subspan_em_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    zero_subspan_em_results,
                    json_file,
                    ensure_ascii=False,
                    indent=4)
        elif args.eval_dataset == "eli5":
            text_column = output_list
            sub_answers = get_sub_answers(args.eval_dataset, ground_truth_file)

            assert len(text_column) == len(
                sub_answers), "The lengths of the columns do not match."

            data = []
            for text, sub_answer in zip(text_column, sub_answers):
                data.append({
                    'predicted_answer': text,
                    'org_answer': [''],
                    'sub_answer': sub_answer
                })
            metric = em_and_subem(data, args.eval_dataset)
            zero_subspan_em_indices = get_zero_subspan_em_indices(
                data, args.eval_dataset)
            zero_subspan_em_results = [result[idx]
                                       for idx in zero_subspan_em_indices]
            zero_subspan_em_json_path = os.path.join(
                args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_zero_subspan_em.json")
            with open(zero_subspan_em_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    zero_subspan_em_results,
                    json_file,
                    ensure_ascii=False,
                    indent=4)
        elif args.eval_dataset == "nqswap":
            ds = load_dataset("pminervini/NQ-Swap")
            org_answers = ds['dev']['org_answer']
            sub_answers = get_sub_answers(args.eval_dataset, ground_truth_file)

            text_column = output_list

            data = []
            for text, org_answer, sub_answer in zip(
                    text_column, org_answers, sub_answers):
                data.append({
                    'predicted_answer': text,
                    'org_answer': org_answer,
                    'sub_answer': sub_answer
                })
            metric = em_and_subem(data, args.eval_dataset)
            zero_subspan_em_indices = get_zero_subspan_em_indices(
                data, args.eval_dataset)
            zero_subspan_em_results = [result[idx]
                                       for idx in zero_subspan_em_indices]
            zero_subspan_em_json_path = os.path.join(
                args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_zero_subspan_em.json")
            with open(zero_subspan_em_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    zero_subspan_em_results,
                    json_file,
                    ensure_ascii=False,
                    indent=4)

        elif args.eval_dataset == "xsum":
            gold_summarys = []
            documents = []
            with open(ground_truth_file, "r") as f:
                for line in f:
                    gold_summarys.append(json.loads(line.strip())['summary'])
                    documents.append(json.loads(line.strip())['document'])
            all_results = []
            num = 0
            for prediction, document, gold_summary in zip(
                    output_list, documents, gold_summarys):
                completion = prediction

                all_refs = [gold_summary]

                # ROUGE-N
                rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
                # ROUGE-1
                rouge1_scores = [score["rouge1"] for score in rouge_scores]
                # ROUGE-2
                rouge2_scores = [score["rouge2"] for score in rouge_scores]
                # ROUGE-L
                rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
                print("rouge done", num)
                factkb_tokenizer = AutoTokenizer.from_pretrained(
                    "roberta-base", padding="max_length", truncation=True
                )
                factkb_model = AutoModelForSequenceClassification.from_pretrained(
                    "bunsenfeng/FactKB", num_labels=2, device_map="auto")
                input_factkb = [[completion, document]]
                factkb_tokens = factkb_tokenizer(
                    input_factkb,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True).to(
                    factkb_model.device)
                factkb_logits = factkb_model(**factkb_tokens).logits
                factkb_res = torch.softmax(factkb_logits, dim=1)
                print("factkb done", num)
                from evaluate import load

                bert_score = load("bertscore")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                bert_score_res = bert_score.compute(
                    predictions=[completion],
                    references=[gold_summary],
                    model_type="microsoft/deberta-xlarge-mnli",
                    lang="en",
                    device=device
                )
                print("bertscore done", num)
                num += 1
                res = {
                    "rouge1": rouge1_scores[0],
                    "rouge2": rouge2_scores[0],
                    "rougeL": rougeL_scores[0],
                    "factKB": float(
                        factkb_res[0][1]),
                    "bertscore_precision": float(
                        bert_score_res["precision"][0]),
                    "bertscore_recall": float(
                        bert_score_res["recall"][0]),
                    "bertscore_f1": float(
                        bert_score_res["f1"][0]),
                }
                all_results.append(res)

            metric = {key: np.mean([res[key] for res in all_results])
                      for key in all_results[0]}
            with open(metric_save_file_path, 'w') as json_file:
                json.dump(metric, json_file, indent=4)
                return
        all_metrics.append(metric)
        with open(metric_save_file_path, 'w') as json_file:
            json.dump(metric, json_file, indent=4)

    if args.eval_dataset == 'eli5':
        final_metric = {
            key: {
                key1: np.mean([metric[key][key1] for metric in all_metrics])
                for key1 in all_metrics[0][key]
            }
            for key in all_metrics[0]
        }
        final_metric.update({
            f"{key}_{key1}_variance": np.var([metric[key][key1] for metric in all_metrics])
            for key in all_metrics[0]
            for key1 in all_metrics[0][key]
        })
    else:
        final_metric = {key: np.mean(
            [metric[key] for metric in all_metrics]) for key in all_metrics[0]}
        final_metric.update({f"{key}_variance": np.var(
            [res[key] for res in all_metrics]) for key in all_metrics[0]})
    print(final_metric)
    metric_save_file_path = os.path.join(
        args.output_folder, f"{args.eval_dataset}_{args.model_id.split('/')[-1]}_metrics_final.json")
    with open(metric_save_file_path, 'w') as json_file:
        json.dump(final_metric, json_file, indent=4)


if __name__ == "__main__":
    main()
