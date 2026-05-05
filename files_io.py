import json
import jsonlines
import os
import pickle as pkl

def read_jsonl(file):
	with jsonlines.open(file,'r') as reader:
		data = list(reader)
	return data

def write_jsonl(item,file):
	os.makedirs(os.path.dirname(file),exist_ok=True)
	with jsonlines.open(file,'a') as writer:
		writer.write(item)

def write_json(data, file, indent=4):
	os.makedirs(os.path.dirname(file), exist_ok=True)
	with open(file,'w',encoding='utf-8') as writer:
		json.dump(data, writer, ensure_ascii=False, indent=indent)

def read_json(file):
	with open(file,'r',encoding='utf-8') as reader:
		data = json.load(reader)
	return data

def write_txt(item, file):
	os.makedirs(os.path.dirname(file), exist_ok=True)
	with open(file,'a',encoding='utf-8') as writer:
		writer.write(item+"\n")


def read_txt(file):
	with open(file,'r',encoding='utf-8') as reader:
		return reader.readlines()


def write_pickle(data, file):
	os.makedirs(os.path.dirname(file), exist_ok=True)
	with open(file,'wb') as writer:
		pkl.dump(data, writer)


def read_pickle(file):
	with open(file,'rb') as reader:
		return pkl.load(reader)