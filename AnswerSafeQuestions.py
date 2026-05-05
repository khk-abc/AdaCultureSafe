import sys
sys.path.append("..")

from utils.files_io import read_json,write_json
from models.APIModel import ApiModel
from glob import glob
from tqdm import tqdm
import numpy as np
import os
from copy import deepcopy
import rich

import argparse
import json
import openai
import LLM_ACCESS_CONFIGS


def get_message(question,elicit_culture=False,true_culture=None):
    if elicit_culture:
        message = [
            {
                "role": 'user',
                "content": f"""{question} PERFORM THE PROCESS OF <culture_knowledge>!"""
            }
        ]
    elif true_culture is not None:
        message = [
            {
                "role": 'user',
                "content": f"""Cultural knowledge: {true_culture}
{question}"""
            }
        ]
    else:
        message = [
            {
                "role": 'user',
                "content": f"""{question}"""
            }
        ]

    return message


def answer_safe_question(llm,question,elicit_culture=False, true_culture=None):
    message = get_message(question=question,elicit_culture=elicit_culture, true_culture=true_culture)
    response,_ = llm.answer(messages=message,run_params={'logprobs':False})

    return response


def answer_single_file(llm, input_file, output_file, elicit_culture=False, import_culture=False):
    results = {}
    if os.path.exists(output_file):
        results = read_json(output_file)

    dataset = read_json(input_file)
    file_info = os.path.basename(input_file).split(".")[0]
    for item_index in tqdm(dataset, desc=f"answering {file_info}..."):
        custom = dataset[item_index]['custom']
        location = dataset[item_index]['location']
        questions = dataset[item_index]['questions']
        if item_index not in results and questions!='FAILED' or any([qa['answer']=='FAILED' for qa in results[item_index]['questions']]):
            questions_with_answer = []
            for qa in questions:
                try:
                    question = " ".join([qa['scene'],qa['requirement']])
                    true_culture = custom if import_culture else None
                    answer = answer_safe_question(llm=llm,question=question, elicit_culture=elicit_culture, true_culture=true_culture)
                    copy_qa = deepcopy(qa)
                    copy_qa.update({"answer": answer})
                    questions_with_answer.append(copy_qa)
                except json.JSONDecodeError as e:
                    print(item_index)
                    print(qa)
                    answer = "FAILED"
                    copy_qa = deepcopy(qa)
                    copy_qa.update({"answer": answer})
                    questions_with_answer.append(copy_qa)
                except openai.BadRequestError as e:
                    print(e)
                    print(item_index)
                    print(qa)
                    answer = "FAILED"
                    copy_qa = deepcopy(qa)
                    copy_qa.update({"answer": answer})
                    questions_with_answer.append(copy_qa)


            res_item = deepcopy(dataset[item_index])
            res_item['questions'] = questions_with_answer

            results[item_index]=res_item

            write_json(data=results, file=output_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",required=True)
    parser.add_argument("--port",default=None,type=str)
    parser.add_argument("--elicit_culture",default=False,action='store_true')
    parser.add_argument("--import_culture",default=False,action='store_true')
    parser.add_argument("--location",default="all",type=str)
    args = parser.parse_args()

    return args


model_type_mapper = {
    "gpt-4o": {
        "api": "",
        "base_url": ""
    },
    "gpt-4o-mini": {
        "api": "",
        "base_url": ""
    },
    "qwen3-max": {
        "api": "",
        "base_url": ""
    },
    "llama2:7b-chat": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen2.5:7b":{
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen2.5:14b": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen2.5:32b": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "llama3.1:8b": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "mistral:7b": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen3:8b": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen2.5:7b-sft": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
    "qwen2.5:7b-dpo": {
        "api": "ollama",
        "base_url": "http://localhost:11434/v1"
    },
}

def main():
    args = get_args()

    llm = ApiModel(model_type=f"<api>{args.model_type}",
                   base_url=model_type_mapper[args.model_type]['base_url'] if args.port is None else f"http://0.0.0.0:{args.port}/v1",
                   api_key=model_type_mapper[args.model_type]['api'] if args.port is None else ""
                   )

    safe_dataset_dir = ""
    safe_files = glob(os.path.join(safe_dataset_dir, "*.json"))
    for file in sorted(safe_files):
        if args.location in file.lower() or args.location == 'all':
            save_path = os.path.join(safe_dataset_dir,f'answers_with_{args.model_type}',os.path.basename(file))
            answer_single_file(llm=llm,input_file=file,output_file=save_path,
                               elicit_culture = args.elicit_culture, import_culture=args.import_culture)

if __name__ == "__main__":
    main()
