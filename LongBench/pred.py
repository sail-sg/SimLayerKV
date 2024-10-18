import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoConfig,LlamaConfig, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "llama3-8b-chat", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "mistral-7b-chat","qwen2-3b-chat",'longchat-7b','yi-9b-chat'])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--type',  type=str, default='vanilla')
    parser.add_argument('--window_size',  type=float, default=1024)
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    device = torch.device(f'cuda:{rank}')
    if (('contrastive' in out_path) or ('stream' in out_path) or ('try' in out_path)or ('minicache' in out_path)) & ('kivi' not in out_path):
        from transformers.models.llama import modeling_llama
        from transformers import GenerationMixin
        from SimLayerKV_llama import _sample,LlamaDecoderLayer,LlamaModel,LlamaForCausalLM
        from SimLayerKV_qwen import Qwen2DecoderLayer,Qwen2Model,Qwen2ForCausalLM
        modeling_llama.LlamaDecoderLayer = LlamaDecoderLayer
        if ('minicache' in out_path):
            modeling_llama.LlamaModel.forward = LlamaModel.minicache_forward
        else:
            modeling_llama.LlamaModel.forward = LlamaModel.forward
        modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM.forward
        from transformers.models.qwen2 import modeling_qwen2
        modeling_qwen2.Qwen2DecoderLayer.forward  = Qwen2DecoderLayer.forward 
        modeling_qwen2.Qwen2Model.forward = Qwen2Model.forward
        modeling_qwen2.Qwen2ForCausalLM.forward = Qwen2ForCausalLM.forward
        from SimLayerKV_stream_attention_qwen import attn_forward
        modeling_qwen2.QWEN2_ATTENTION_CLASSES["flash_attention_2"].forward = attn_forward
        modeling_qwen2.QWEN2_ATTENTION_CLASSES["eager"].forward = attn_forward
        modeling_qwen2.QWEN2_ATTENTION_CLASSES["sdpa"].forward = attn_forward
        GenerationMixin._sample = _sample
    if 'snapkv' in out_path:
        from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral, replace_qwen
        compress_args = json.load(open('config/ablation_c1024_w32_k7_maxpool.json', "r"))
        compress = True
        replace_llama()
        replace_qwen()
        replace_mistral()
        replace_mixtral()
    else:
        compress = False
        compress_args = None
    if 'h2o' in out_path:
        from utils_real_drop.modify_llama import H2OLlamaForCausalLM
        from utils_real_drop.modify_qwen import H2OQwen2ForCausalLM
        config = LlamaConfig.from_pretrained(model2path[model_name])
        tokenizer = LlamaTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
        config.hh_size = 4
        config.recent_size = 4096
        if 'qwen' not in model_name:
            model = H2OLlamaForCausalLM.from_pretrained(model2path[model_name], config=config)
        else:
            model = H2OQwen2ForCausalLM.from_pretrained(model2path[model_name], config=config)
        model = model.half().to(device).eval()
    else:   
        model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device,compress,out_path)

    try:
        with open(out_path, 'r') as file:
            data_lines = file.readlines()
        modified_data = []
        for line in data_lines:
            # Parse the line as a JSON object
            d = json.loads(line)
            d_ = {"answers": d["answers"], "all_classes": d["all_classes"], "length": d["length"]}
            modified_data.append(d_)
    except:
        modified_data = []  
    
    for json_obj in tqdm(data):
        # print(json_obj)
        if {"answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]} in modified_data:
            continue
        ############################################################################################################
        # load compress args
        if compress == True:
            window_sizes = None
            max_capacity_prompts = None
            kernel_sizes = None
            pooling = None
            if 'window_sizes' in compress_args:
                window_sizes = compress_args['window_sizes']
            if 'max_capacity_prompts' in compress_args:
                max_capacity_prompts = compress_args['max_capacity_prompts']
            if 'kernel_sizes' in compress_args:
                kernel_sizes = compress_args['kernel_sizes']
            if 'pooling' in compress_args:
                pooling = compress_args['pooling']

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                if i >= len(max_capacity_prompts):
                    model.model.layers[i].self_attn.config.max_capacity_prompt = 768
                else:
                    model.model.layers[i].self_attn.config.max_capacity_prompt = 768
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling
        ############################################################################################################
        
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        if 'llama3' in out_path:
            threshold = 0.9
        elif 'llama2' in out_path:
            threshold = 0.65
        elif 'mistral' in out_path:
            threshold = 0.8
        elif 'qwen' in out_path:
            threshold = 0.85
        

        if 'stream' in out_path:
            threshold = 0
        if 'minicache' in out_path:
            threshold = -1
        if 'kivi_2bit' in out_path:
            threshold = 1
        
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                pad_token_id = tokenizer.eos_token_id,
                threshold = threshold
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id = tokenizer.eos_token_id,
                threshold = threshold
            )[0]
        if 'h2o' in str(out_path):
            from utils_real_drop.modify_llama import H2OLlamaAttention
            from utils_real_drop.modify_qwen import H2OQwen2Attention
            for name, m in model.named_modules():
                if isinstance(m, H2OLlamaAttention):
                    m._clean_cache()
                if isinstance(m, H2OQwen2Attention):
                    m._clean_cache()
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device,compress,out_path = ''):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name  or "long" in model_name or "yi" in model_name :
        tokenizer = LlamaTokenizer.from_pretrained(path, trust_remote_code=True)
        model = LlamaForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attention_2=compress).to(device)
    elif ("llama" in model_name) or ('mistral' in model_name):

        # replace_llama_attn_with_flash_attn()
        if 'llama' in out_path:
            tokenizer = AutoTokenizer.from_pretrained(path)
        else:

            tokenizer = LlamaTokenizer.from_pretrained(path, trust_remote_code=True)
        if 'kivi' in out_path:
            config = LlamaConfig.from_pretrained(path)
            if '2bit' in out_path:
                config.k_bits = 2 # current support 2/4 bit for KV Cache
                config.v_bits = 2 # current support 2/4 bit for KV Cache
                from KIVI.models.llama_kivi_2bit import LlamaForCausalLM_KIVI
            else:
                from KIVI.models.llama_kivi import LlamaForCausalLM_KIVI,greedy_search,LlamaAttention_KIVI,LlamaFlashAttention_KIVI
                from transformers import GenerationMixin
                GenerationMixin.greedy_search = greedy_search
                
                config.k_bits = 4 # current support 2/4 bit for KV Cache
                config.v_bits = 4 # current support 2/4 bit for KV Cache
                
            config.group_size = 32
            config.residual_length = 32 # the number of recent fp16 tokens
            # if 'llama3' in out_path:
            config.use_flash = True
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                # device_map="auto",
            ).to(device)
            
        else:
            model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, use_flash_attention_2=compress).to(device)
        if 'llama3' in model_name:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    elif 'qwen' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

        # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=compress
        ).to(device).eval()

    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        print(dataset)
        
        if args.window_size == 512:
            if '512' not in args.type:
                args.type = args.type+str(args.window_size)
        elif args.window_size != 1024:
            if str(args.window_size) not in args.type:
                args.type = args.type+'+'+str(args.window_size)
        
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{write_model_name}"):
                os.makedirs(f"pred_e/{write_model_name}")
            out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred_e/{write_model_name}"):
                os.makedirs(f"pred_e/{write_model_name}")
            out_path = f"pred_e/{write_model_name}/{dataset}.jsonl"
          
        if not os.path.exists(f"pred/{model_name}/{args.type}"):
            os.makedirs(f"pred/{model_name}/{args.type}")
            out_path = f"pred/{model_name}/{args.type}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
