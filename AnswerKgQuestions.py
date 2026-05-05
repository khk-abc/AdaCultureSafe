import sys

sys.path.append('..')
import json
from utils.files_io import read_json,write_json
from models.APIModel import ApiModel
from glob import glob
from tqdm import tqdm
import numpy as np
from rich import print as rprint
import json_repair
import os
import argparse
from copy import deepcopy
import re
import openai
import LLM_ACCESS_CONFIGS



def get_message(question,options,culture=False):
    str_options = []
    for op_index, op_content in options.items():
        str_options.append(f"""({op_index}) {op_content}""")
    str_options = "\n".join(str_options)
    if culture:
        message = [
            {
                "role": 'system',
                "content": """Please choose the correct option from all the given options for the question!

                    NOTE: YOU FIRST BEGIN TO PERFORM THE PROCESS OF <cultural knowledge>, AND THEN OUTPUT YOUR CHOICE.
                    
                    NOTE: YOU MUST OUTPUT YOUR CHOICE WITH THE FOLLOWING JSON FORMAT: 
                    {"answer": "ONLY PLACE THE OPTION INDEX OF YOUR SELECTED OPTION HERE LIKE: A, B, C, D..."}
                    Example: <culture_knowledge>\n THE CONTENT OF culture_knowledge. \n</culture_knowledge>\n{"answer": "A"}

                    NOTE: If there are many correct options, you can select multiple options, such as {"answer": "B/C"}

                    NOTE: YOU MUST ONLY OUTPUT THE OPTION INDEX AND NOT OUTPUT THE DETAILS OF THE OPTION.""",
            },
            {
                "role": 'user',
                "content": f"""<QUESTION> {question} </QUESTION>
                    <OPTIONS> {str_options} </OPTIONS>
                    <culture_knowledge>\n"""
            }
        ]
    else:
        message = [
            {
                "role": 'system',
                "content": """Please choose the correct option from all the given options for the question!
    
                    NOTE: YOU MUST OUTPUT WITH THE FOLLOWING JSON FORMAT: 
                    {"answer": "ONLY PLACE THE OPTION INDEX OF YOUR SELECTED OPTION HERE: A, B, C, D..."}
                    Example: {"answer": "A"}
    
                    NOTE: If there are many correct options, you can select multiple options, such as {"answer": "B/C"}
    
                    NOTE: YOU MUST ONLY OUTPUT THE OPTION INDEX AND NOT OUTPUT THE DETAILS OF THE OPTION.""",
            },
            {
                "role": 'user',
                "content": f"""<QUESTION> {question} </QUESTION>
                    <OPTIONS> {str_options} </OPTIONS>"""
            }
        ]

    return message

def answer_kg_question(llm,question,options,culture=False):
    message = get_message(question=question,options=options,culture=culture)
    response,_ = llm.answer(messages=message,run_params={'logprobs':False})

    return response


def answer_single_file(llm,file,save_path,culture=False):
    results = {}
    if os.path.exists(save_path):
        results = read_json(save_path)

    dataset = read_json(file)
    base_info = os.path.basename(file).split(".")[0]
    for item_index in tqdm(dataset, desc=f"answering {base_info}..."):
        custom = dataset[item_index]['custom']
        location = dataset[item_index]['location']
        questions = dataset[item_index]['questions']
        if item_index not in results and questions!='FAILED' or any([qa['answer']=='FAILED' for qa in results[item_index]['questions']]):
            correct_num = 0
            questions_with_answer = []
            for qa in questions:
                copy_qa = deepcopy(qa)
                try:
                    raw_answer = answer_kg_question(llm=llm, question=qa['question'], options=qa['options'],culture=culture)
                    print(raw_answer)
                    pattern1 = r'\{[^}]+\}'
                    answer = re.findall(pattern=pattern1, string=raw_answer)[-1]
                    print(answer)

                    answer = json.loads(json_repair.repair_json(answer))['answer']
                    ground_truth = qa['label']
                    if ground_truth.lower() in answer.lower():
                        correct_num += 1
                except json.JSONDecodeError as e:
                    print(item_index)
                    print(qa)
                    answer = "FAILED"
                except openai.BadRequestError as e:
                    print(item_index)
                    print(qa)
                    answer = "FAILED"

                copy_qa.update({"answer": answer})
                questions_with_answer.append(copy_qa)
                rprint(questions_with_answer[-1])

            item_acc = correct_num / len(questions)

            res_item = deepcopy(dataset[item_index])
            res_item['questions'] = questions_with_answer
            res_item['accuracy'] = item_acc

            rprint(res_item)

            results[item_index] = res_item

            write_json(data=results, file=save_path)

            total_acc = np.mean([results[index]['accuracy'] for index in results])
            print(f"{location}: {total_acc}")

    total_acc = np.mean([results[index]['accuracy'] for index in results])
    print(f"{location}: {total_acc}")



model_type_mapper = {
    "gpt-4o": {
        "api":"",
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
    model_type = args.model_type

    llm = ApiModel(model_type=f"<api>{args.model_type}",
                   base_url=model_type_mapper[args.model_type]['base_url'] if args.port is None else f"http://0.0.0.0:{args.port}/v1",
                   api_key=model_type_mapper[args.model_type]['api'] if args.port is None else "",
                   )


    kg_dataset_dir = ""
    kg_files = glob(os.path.join(kg_dataset_dir,"*.json"))

    for file in sorted(kg_files):
        if args.location in file.lower() or args.location == 'all':
            save_path = os.path.join(kg_dataset_dir,f'answers_with_{model_type}',os.path.basename(file))
            answer_single_file(llm=llm,file=file,save_path=save_path,culture=args.culture)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",required=True)
    parser.add_argument("--port",default=None,type=str)
    parser.add_argument("--culture",default=False,action='store_true')
    parser.add_argument("--location", default="all", type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
