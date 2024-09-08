import os
import random
import json
import argparse

import re 
import string
import collections
from collections import Counter
from time import sleep

from nltk.corpus import stopwords
import numpy as np
stops = set(stopwords.words('english'))
puncs = list(string.punctuation)
from tqdm import trange, tqdm

import torch
from torch import tensor
from torchmetrics.classification import BinaryCalibrationError
from vllm import LLM, SamplingParams
from huggingface_hub import login

# import openai
import pandas as pd
import getpass
from datasets import load_dataset

login(token="your_token_here")


def load_strategyqa(filename):
    # label: true, false
    data = []
    with open(filename, "r",encoding="utf-8") as f:
        raw_data = json.load(f)
    
    mapping = {"True":"Yes","False":"No"}
    
    for item in raw_data:
        temp = {}
        temp["question"] = item["question"]
        temp["answers"] = mapping[str(item["answer"])]
        temp["facts"] = "\n".join(item["facts"])
        temp["decomposition"] = item["decomposition"]
        data.append(temp)
        
    return data


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(puncs)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) in normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)  
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def answer_match_textqa(pred, ans):
    pred = answer_extract_textqa(pred)
    return normalize_answer(pred) == normalize_answer(ans)


def answer_extract_textqa(pred):
    prefix = "answer is "
    if prefix in pred:
        idx = pred.rfind(prefix)
        # print ("extracted ans string: ", pred[idx + len(prefix) : ])
        return pred[idx + len(prefix) : ]
    return pred.strip()


def single_ans_em(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) !=list:
        gold = [gold]
    pred = answer_extract_textqa(pred)
    return max(compute_exact(pred, a) for a in gold)

def single_ans_f1(pred, gold):
    # pred: prediction string
    # gold: a list of gold answer strings
    if type(gold) !=list:
        gold = [gold]
    pred = answer_extract_textqa(pred)
    return max(compute_f1(pred, a) for a in gold)


def prompt_decoration(example,task="odqa"):
    # naive QA prompt decoration        
    if "boolq" in task or "odqa" in task:
        
        if "odqa" in task:
            hint = "Answer in one phrase."
        elif "boolq" in task:
            hint = "Yes or No? Choose one answer."
        
        template= "Question: [Q] "+ hint +"\nAnswer:"
        prompt = template.replace("[Q]",example["question"])
 
        if "-ptrue" in task:
            template = example["prompt"].strip().replace("\n Answer:", "")
            template += """
            Possible Answer: [A]
            Is the possible answer:
                (A) True
                (B) False
            The possible answer is: """
            template = template.replace("[H]",hint)
            prompt = template.replace("[Q]",example["question"]).replace("[A]", example["pred"])
            return prompt
        
        if "-verbalized" in task:
            template = example["prompt"].strip() + example["pred"]
            template += "\nConfidence(0-100): "
            return template
        
        if "-knowledge" in task:
            # imported from paper: https://arxiv.org/pdf/2110.08387.pdf
            # detailed prompt used: https://www.promptingguide.ai/techniques/knowledge
            ori_maxlen = 0+args.maxlen
            args.maxlen = 80
            prompt="""Question: [Q] \n Generate some knowledge about the concepts in the question?  \nKnowledge: """
            prompt = prompt.replace("[Q]",example["question"])
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt = prompt.replace("\n Generate some knowledge about the concepts in the question?","")
            prompt += ans
            if "-explain" in task:
                prompt += "\n Explain and Answer, "+ hint +"\nAnswer:"
            else:
                prompt += "\n\n Therefore, [Q] [H] \nAnswer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            
        if "-tot" in task: 
            # this is a simluated version of the search-based ToT: https://github.com/dave1010/tree-of-thought-prompting/tree/main
            # for the original ToT, see: https://arxiv.org/abs/2305.10601
            ori_maxlen = 0+args.maxlen
            args.maxlen = 240
            
            # template 1
            # prompt = "Imagine three different experts are answering this question.\n"
            # prompt += "All experts will write down 1 step of their thinking, then share it with the group.\n"
            # prompt += "Then all experts will go on to the next step, etc.\n"
            # prompt += " If any expert realises they're wrong at any point then they leave.\n"
            
            # template 2
            # prompt = "Simulate three brilliant, logical experts collaboratively answering a question.\
            # Each one verbosely explains their thought process in real-time, \
            # considering the prior explanations of others and openly acknowledging mistakes. \
            # At each step, whenever possible, each expert refines and builds upon the thoughts of others, \
            # acknowledging their contributions. They continue until there is a definitive answer to the question. \
            # For clarity, your entire response should be in a markdown table."
            
            # template 3
            prompt = """
            Identify and behave as three different experts that are appropriate to answering this question.
            All experts will write down the step and their thinking about the step, then share it with the group.
            Then, all experts will go on to the next step, etc.
            At each step all experts will score their peers response between 1 and 5, 1 meaning it is highly unlikely, and 5 meaning it is highly likely.
            If any expert is judged to be wrong at any point then they leave.
            After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus solution or your best guess solution.
            """
            prompt = prompt.replace("\t","")
            prompt += "The question is: [Q]\n"
            
            prompt = prompt.replace("[Q]", example["question"])
            
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += ans #Choose one.
            prompt += "\n\n Therefore, [Q] [H] \nAnswer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            args.maxlen = ori_maxlen

        if "-selffacts" in task:
            ori_maxlen = 0+args.maxlen
            args.maxlen = 80
            prompt="""Question: [Q] \n What are the facts needed to answer this question?  \nSupporting Facts: """
            prompt = prompt.replace("[Q]",example["question"])
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += "\n [F]"
            prompt = prompt.replace("[F]",ans)
            prompt += "\n\n Therefore, [Q] [H] \n Answer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            args.maxlen = ori_maxlen
            
        if "-facts" in task:
            ori_maxlen = 0+args.maxlen
            args.maxlen = 80
            prompt="""Question: [Q] \n What are the facts needed to answer this question?  \nSupporting Facts: """
            prompt = prompt.replace("[Q]",example["question"])
            prompt += "\n [F]"
            prompt = prompt.replace("[F]",example["facts"])
            prompt += "\n\n Therefore, [Q] [H] \n Answer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            args.maxlen = ori_maxlen
            
        if "-far_human" in task:
            ori_maxlen = 0+args.maxlen
            args.maxlen = 80
            prompt="""Question: [Q] \n What are the facts needed to answer this question?  \nSupporting Facts: """
            prompt = prompt.replace("[Q]",example["question"])
            prompt += "\n [F]"
            prompt = prompt.replace("[F]",example["facts"])
            
            prompt += "\n Given above facts you provided, what is your reasoning? \nReasoning: \n"
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += "[R]"
            
            prompt += "\n\n Therefore, [Q] [H] \n Answer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            args.maxlen = ori_maxlen
            
        if "-gold" in task:
            ori_maxlen = 0+args.maxlen
            args.maxlen = 80
            prompt="""Question: [Q] \n What are the facts needed to answer this question? \nFollow-up Questions: [D] \nSupporting Facts: [F]"""
            prompt = prompt.replace("[Q]",example["question"])
            prompt = prompt.replace("[D]","\n".join(example["decomposition"]))
            prompt = prompt.replace("[F]",example["facts"])
            
            prompt += "\n\n Therefore, [Q] [H] \nAnswer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            args.maxlen = ori_maxlen
        
        if "-cot" in task:
            prompt="""Question: [Q] \nLet's think step-by-step: """
            prompt=prompt.replace("[Q]", example["question"])

            # ori_maxlen = 0+args.maxlen
            # args.maxlen = 80
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            # args.maxlen = ori_maxlen

            prompt += " \n[T]" # T can be fixed cot extracted elsewhere or written by human
            prompt = prompt.replace("[T]", ans.strip().replace("Answer:",""))
            prompt += "\n\n Therefore, [Q] [H] \nAnswer: "
            prompt = prompt.replace("[H]",hint)
            prompt=prompt.replace("[Q]", example["question"])
            return prompt

        elif "-far" in task:
            # ori_maxlen = 0+args.maxlen
            # args.maxlen = 80
            prompt="""Question: [Q] \nWhat are the facts needed to answer this question?  \nSupporting Facts: """
            prompt = prompt.replace("[Q]",example["question"])
            
            # fact step
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += "\n [F]"
            prompt = prompt.replace("[F]",ans.strip())
            # source step
            prompt += "\n Given above facts you provided, what are their sources? \nSources: \n"
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += "[S]"
            prompt = prompt.replace("[S]",ans.strip())
            # reason step
            prompt += "\n Given above facts you provided, what is your reasoning? \nReasoning: \n"
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            prompt += "[R]"
            prompt = prompt.replace("[R]",ans.strip())
            
        
            # prompt += "\n\n Therefore, [Q] Explain and Answer, [H] \nAnswer: "
            prompt += "\n\n Therefore, [Q] [H] Choose one.\n Answer: "
            prompt = prompt.replace("[H]",hint)
            prompt = prompt.replace("[Q]", example["question"])
            # args.maxlen = ori_maxlen
        
        elif "-selfask" in task:
            prompt="""Question: [Q] \nAre follow up questions needed here, Yes or No: """
            prompt = prompt.replace("[Q]",example["question"])
            ans,_,_,_ = get_one_step_prompt(args, prompt)
            if "yes" in ans.lower():
                
                prompt += ans
                ori_maxlen = 0+args.maxlen
                args.maxlen = 150
                prompt += "\nFollow up questions: "
                ans,_,_,_ = get_one_step_prompt(args, prompt)
                
                if "-step" in task:
                    # step by step question asking
                    subqs = ans.split("?")
                    for subq in subqs[:3]:
                        prompt += subq.strip() +"? Answer in one sentence. Answer:"
                        args.maxlen = 50
                        ans,_,_,_ = get_one_step_prompt(args, prompt)
                        prompt += ans +"\n"
                else:
                    prompt += ans
                    prompt += "\n Intermediate answers to follow up questions: "
                    ans,_,_,_ = get_one_step_prompt(args, prompt)
                    prompt += ans
                
                prompt += "Therefore, [Q] the final answer is, "+hint+"\nAnswer:"
                args.maxlen = ori_maxlen
                
            prompt = prompt.replace("[Q]",example["question"])
        
        elif "-decomposed" in task:
            # works only for sqa for now
            prompt="Question: [Q]"
            prompt = prompt.replace("[Q]",example["question"])
            
            if "-step" in task:
                # step by step question asking
                for i, subq in enumerate(example["decomposition"][:4]):
                    prompt += "\n Q" + str(i+1) + ":" + subq
                    ans,_,_,_ = get_one_step_prompt(args, prompt)
                    prompt += "\n" + ans
            else:
                prompt += "\n Follow up questions: "
                for i, subq in enumerate(example["decomposition"][:4]):
                    prompt += "\n Q" + str(i+1) + ": " + subq
                prompt += "\n Answer to these follow up quesitions:"
                ans,_,_,_ = get_one_step_prompt(args, prompt)
                prompt += "\n" + ans
                
            prompt += "\n Therefore, [Q] the final answer is, "+hint+"\nAnswer:"
            prompt = prompt.replace("[Q]",example["question"])
        
        return prompt
    
    print("nothing matched")
    return False


def get_one_step_prompt(args, prompt):
    
    tokenized = tokenizer.tokenize(prompt)
    
    if len(tokenized) >= 2040:
        print ("len tokenized: ", len(tokenized))
        prompt = tokenizer.decode(tokenized[-2040 : ])
        
    # no batch inference for now
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

    output = outputs[0]

    prompt = output.prompt
    generated_text = output.outputs[0].text
    tokens = output.outputs[0].token_ids
    
    logprobs = []
    for i, tok_prob in enumerate(output.outputs[0].logprobs):
        logprobs.append(tok_prob[tokens[i]].logprob)
        
    # top_probs, top_tokens    
    perplexity = np.exp((np.mean(logprobs)))
    
    return generated_text.strip(), perplexity, tokens, logprobs


def perplexity_calibration(args, dataset):
    
    collection = []
    perplexities = []
    accs = []
    
    for example in tqdm(dataset):
        # print(example)
        prompt = prompt_decoration(example,task=args.prompt_type)
        # print(prompt)
        output, perplexity, top_tokens, top_probs = get_one_step_prompt(args, prompt)
        
        acc = 0
        answer_key = "answers"
        if "odqa" in args.prompt_type:
            for gold_ans in example["answers"]:
                if compute_exact(gold_ans,output):
                    acc = 1
        elif "boolq" in args.prompt_type:
            if example["answers"] in output:
                acc = 1
                
        elif "humaneval" in args.prompt_type:
            acc += compute_f1(example["output"],output)
            answer_key = "output"
        
        # nan issue
        if perplexity != perplexity:
            perplexity = 10 # a ad-hoc large value
        
        perplexities.append(perplexity)
        accs.append(acc)
        
        if args.print_prompt:
            print(prompt,"\n", output,"|", example[answer_key], acc, "|", perplexity)
        
        collection.append([output, acc, perplexity,top_tokens,top_probs, prompt])

    print("=====    Perplexity Calibration   ========")
    print("Perlexity:", sum(perplexities)/len(perplexities))
    # print("accs=",end="")
    # print(accs)
    # print("perplexities=",end="")
    # print(perplexities)
    
    if "humaneval" not in args.prompt_type:
        print("Acc:", sum(accs)/len(accs))
        metric = BinaryCalibrationError(n_bins=2, norm='l1')
        ece = metric(torch.tensor(perplexities), torch.tensor(accs)).tolist()
        print("ECE:", ece)
    else:
        print("F1:", sum(accs)/len(accs))
    macroce = macroce_post_calibration(accs, perplexities)
    print("MacroCE", macroce)
    
    overall_collection = {}
    overall_collection["perplexity"] = collection
    overall_collection["ptrue"] = []
    overall_collection["verbalized"] = []
    
    if args.add_ptrue:
        args.prompt_type += "-ptrue"
        new_collection = ptrue_post_calibration(args, dataset, collection)
        overall_collection["ptrue"].extend(new_collection)
    
    if args.add_verbalized:
        args.prompt_type += "-verbalized"
        new_collection = verbalized_post_calibration(args, dataset, collection)
        overall_collection["verbalized"].extend(new_collection)
    
    return overall_collection  


def macroce_post_calibration(corrects, confidence):
    # implemented from paper: https://aclanthology.org/2022.findings-emnlp.204.pdf
    # must be used when the metric is acc
    
    ice_pos = []
    ice_neg = []
    
    for i,corr in enumerate(corrects):
        if corr:
            ice_pos.append(confidence[i])
        else:
            ice_neg.append(confidence[i])
    
    if len(ice_pos)==0:
        ice_pos=[0]
    if len(ice_neg)==0:
        ice_neg=[0]
    
    return (sum(ice_pos)/len(ice_pos)+sum(ice_neg)/len(ice_neg))/2


def ptrue_post_calibration(args, dataset, collection):
    # implemented from paper: https://arxiv.org/pdf/2207.05221.pdf
    
    new_collection = []
    ptrues = []
    accs = []
    
    if "ptrue" not in args.prompt_type:
        args.prompt_type = args.prompt_type + "-ptrue"
    
    for i, example in enumerate(tqdm(dataset)):
        # print(example)
        example["pred"] = collection[i][0]
        example["prompt"] = collection[i][-1]
        prompt = prompt_decoration(example,task=args.prompt_type)

        output, perplexity, top_tokens, top_probs = get_one_step_prompt(args, prompt)
        
        ptrue = 0

        a_token = tokenizer.encode("A")[0]
        b_token = tokenizer.encode("B")[0]

        if "A" in output:
            ptrue = np.exp(max(top_probs))
        elif "B" in output:
            ptrue = 1-np.exp(max(top_probs))

        # try:
        #     for j, tok_id in enumerate(top_tokens):
        #         if a_token == tok_id:
        #             ptrue = np.exp(top_probs[j])
        #             break
        #         elif b_token == tok_id:
        #             ptrue = 1-np.exp(top_probs[j])
        #             break
        # except:
        #     print(a_token)
        #     print(top_tokens)
        #     print(top_probs)

        ptrues.append(ptrue)
        accs.append(collection[i][1])
        if args.print_prompt:
            print(prompt,"\n", output,"\n ---------------------")
        new_collection.append([output,ptrue,collection[i][1],top_tokens,top_probs])
    
    # calculate the calibration
    print("=====    Ptrue Calibration   ========")
    print("Ptrue:", sum(ptrues)/len(ptrues))
    # print("ptrues=",end="")
    # print(ptrues)
    
    if "humaneval" not in args.prompt_type:
        # odqa or boolq
        try: 
            metric = BinaryCalibrationError(n_bins=2, norm='l1')
            ece = metric(torch.tensor(ptrues), torch.tensor(accs)).tolist()
            print("Acc:", sum(accs)/len(accs))
            print("ECE:", ece)
        except:
            print(ptrues)
            print(accs)
        
    else:
        print("F1:", sum(accs)/len(accs))
        
    macroce = macroce_post_calibration(accs, ptrues)
    print("MacroCE", macroce)
    
    return new_collection


def verbalized_post_calibration(args, dataset, collection):
    # implemented from paper: https://arxiv.org/pdf/2205.14334.pdf
    
    new_collection = []
    verbals = []
    accs = []
    
    if "verbalized" not in args.prompt_type:
        args.prompt_type = args.prompt_type + "-verbalized"
        
    if "ptrue" in args.prompt_type:
        args.prompt_type = args.prompt_type.replace("ptrue","")
        
    for i, example in enumerate(tqdm(dataset)):
        # print(example)
        example["pred"] = collection[i][0]
        example["prompt"] = collection[i][-1]
        prompt = prompt_decoration(example,task=args.prompt_type)
        output, perplexity, top_tokens, top_probs = get_one_step_prompt(args, prompt)
        output = output.strip()
        # output = output.split(" ")[0]
        output = output.replace("%"," ").replace("."," ").replace(";"," ").replace(","," ").replace("-"," ").replace(":"," ")
        output = output.replace("("," ").replace(")"," ").replace("Confidence"," ")
        for tok in output.split():
            try:
                verbal = float(tok)
                break
            except:
                verbal = 0.0
                # print(output)
           
        verbals.append(verbal)
        accs.append(collection[i][1])
        if args.print_prompt:
            print(prompt,"\n", output,"\n ---------------------")
        new_collection.append([output,verbal,collection[i][1],top_tokens,top_probs])
    
    # calculate the calibration
    print("=====    Verbalized Calibration   ========")
    print("Verbalized Score:", sum(verbals)/len(verbals))
#     print("verbals=",end="")
#     print(verbals)
    
    if "humaneval" not in args.prompt_type:
        # odqa or boolq
        try: 
            metric = BinaryCalibrationError(n_bins=2, norm='l1')
            ece = metric(torch.tensor(verbals), torch.tensor(accs)).tolist()
            print("Acc:", sum(accs)/len(accs))
            print("ECE:", ece)
        except:
            print(ptrues)
            print(accs)
        
    else:
        print("F1:", sum(accs)/len(accs))
        
    macroce = macroce_post_calibration(accs, verbals)
    print("MacroCE", macroce)
    
    return new_collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## parameters
    parser.add_argument("--data", default='sqa', type=str, required=False,
                        help="choose which dataset to use")
    parser.add_argument("--max_len", default=80, type=int, required=False,
                        help="output max length")
    parser.add_argument("--temprature", default=0.8, type=float, required=False,
                        help="llm temperature")  
    parser.add_argument("--download_dir", default="./local_model/", type=str, required=False,
                        help="where to put the downloaded model weights")  
    parser.add_argument("--prompt_type", default="boolq", type=str, required=False,
                        help="boolq for sqa, odqa for webq")  
    parser.add_argument("--model_name_or_path", default="lmsys/vicuna-13b-v1.3", type=str, required=False,
                        help="the desired model")                    
    parser.add_argument("--add_ptrue", default=0, type=int, required=False,
                        help="if we condcut pture experiments")
    parser.add_argument("--add_verbalized", default=0, type=int, required=False,
                        help="if we conduct verbalized confidence experiments")
    parser.add_argument("--print_prompt", default=0, type=int, required=False,
                        help="for debugging")

    args = parser.parse_args()

    # load dataset
    if args.data == "webq":
        dataset = load_dataset("web_questions")
        mini_wq = dataset["test"][:100]
    elif args.data == "sqa":
        sqa = load_strategyqa("./data/strategy_qa/dev.json")

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=args.temprature, 
        logprobs=1, 
        # top_p=0.95, 
        # max_tokens=args.max_len,
        # prompt_logprobs=0,
        )

    # Create an LLM.
    args.model_name_or_path = "lmsys/vicuna-13b-v1.3"
    
    args.model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    
    # args.model_name_or_path = "baichuan-inc/Baichuan2-13B-Chat"

    llm = LLM(model=args.model_name_or_path,
        trust_remote_code=True,
        seed=42,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.7,
        download_dir=args.download_dir,
        max_model_len=2048, 
        )

    tokenizer = llm.get_tokenizer()

    # batched experiments
    for tp in ["","-far"]: #,"-cot","-selfask-step"
        print("Running this prompt type:", tp)
        args.prompt_type += tp
        
        overall_collection = perplexity_calibration(args, sqa)
        
        args.prompt_type = "boolq"+tp
        
        filename = "./output_collection/"+args.prompt_type+"_collection.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(overall_collection, f)
            
        print("================")
        args.prompt_type = "boolq" 