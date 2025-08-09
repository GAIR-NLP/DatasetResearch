# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import random
from typing import Optional

import fire
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def format_few_shot_prompt(original_prompt: str, few_shot_examples: list, few_shot_format: str = "qa") -> str:
    """format few-shot prompt"""
    if not few_shot_examples:
        return original_prompt
    
    few_shot_text = ""
    
    if few_shot_format == "qa":
        # qa format
        for i, example in enumerate(few_shot_examples, 1):
            few_shot_text += f"Example{i}: \n"
            few_shot_text += f"Question: {example['input'].strip()}\n"
            few_shot_text += f"Answer: {example['output'].strip()}\n\n"
    
    elif few_shot_format == "chat":
        # chat format
        for i, example in enumerate(few_shot_examples, 1):
            few_shot_text += f"Chat{i}: \n"
            few_shot_text += f"User: {example['input'].strip()}\n"
            few_shot_text += f"Assistant: {example['output'].strip()}\n\n"
    
    elif few_shot_format == "simple":
        # simple format
        for example in few_shot_examples:
            few_shot_text += f"Input: {example['input'].strip()}\n"
            few_shot_text += f"Output: {example['output'].strip()}\n\n"
    
    # combine prompt
    final_prompt = f"Please refer to the following example: \n\n{few_shot_text} Now answer the following question: {original_prompt}"
    return final_prompt


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 4096,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    # few-shot parameters
    train_dataset: Optional[str] = None,
    n_shot: int = 3,
    few_shot_format: str = "qa",
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: 
    # normal inference (test set)
    python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset test_set
    
    # few-shot inference (test set + train set as few-shot examples)
    python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset test_set --train_dataset train_set --enable_few_shot --n_shot 3
    """
    print("******************************HERE*****************************")
    if train_dataset is None:
        enable_few_shot = False
    else:
        enable_few_shot = True
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": 0.9
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)

    # load test dataset (for inference)
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    test_dataset = dataset_module["train_dataset"]  # here actually is test data
    
    # prepare few-shot examples
    few_shot_examples = []
    if enable_few_shot and n_shot > 0 and train_dataset is not None:
        print(f"🎯 启用Few-shot模式，从训练集 '{train_dataset}' 随机选择 {n_shot} 个示例")
        
        # load train set as few-shot examples

        # create new data_args to load train set
        from copy import deepcopy
        train_data_args = deepcopy(data_args)
        train_data_args.dataset = [train_dataset]
        print(train_data_args)
        train_dataset_module = get_dataset(template_obj, model_args, train_data_args, training_args, "ppo", **tokenizer_module)
        train_data = train_dataset_module["train_dataset"]
        print(f"📚 成功加载训练集，包含 {len(train_data)} 个样本")
        
        # randomly select n_shot examples from train set as few-shot
        if len(train_data) >= n_shot:
            # set random seed to ensure reproducibility
            if seed is not None:
                random.seed(seed)
            
            # randomly select indices
            indices = random.sample(range(len(train_data)), n_shot)
            
            for idx in indices:
                # get original data
                sample = train_data[idx]
                
                # decode input and output
                input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=skip_special_tokens)
                output_text = tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])),
                    skip_special_tokens=skip_special_tokens,
                )
                
                few_shot_examples.append({
                    "input": input_text,
                    "output": output_text
                })
            
            print(f"🎯 successfully selected {len(few_shot_examples)} few-shot examples from train set")
        else:
            print(f"⚠️ train set sample number ({len(train_data)}) is less than requested few-shot number ({n_shot})")
            n_shot = min(len(train_data), n_shot)
            few_shot_examples = []
                

            
    elif enable_few_shot and train_dataset is None:
        print("⚠️ enabled few-shot but no train_dataset specified, will execute without few-shot")
        enable_few_shot = False

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    all_prompts, all_preds, all_labels = [], [], []

    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = test_dataset[i : min(i + batch_size, len(test_dataset))]

        for j in range(len(batch["input_ids"])):
            # get original input
            original_input_ids = batch["input_ids"][j]
            original_prompt = tokenizer.decode(original_input_ids, skip_special_tokens=skip_special_tokens)
            
            # apply few-shot if enabled
            if enable_few_shot and few_shot_examples:
                # format few-shot prompt
                modified_prompt = format_few_shot_prompt(original_prompt, few_shot_examples, few_shot_format)
                
                # re-encode
                modified_input_ids = tokenizer.encode(modified_prompt, add_special_tokens=True)
                
                # check length limit
                if len(modified_input_ids) > cutoff_len:
                    print(f"⚠️ few-shot prompt too long ({len(modified_input_ids)} > {cutoff_len}), truncated")
                    modified_input_ids = modified_input_ids[:cutoff_len]
                
                final_input_ids = modified_input_ids
                final_prompt = modified_prompt
            else:
                final_input_ids = original_input_ids
                final_prompt = original_prompt
            
            # process multi-modal data
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image, image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
            else:
                multi_modal_data = None

            vllm_inputs.append({"prompt_token_ids": final_input_ids, "multi_modal_data": multi_modal_data})
            prompts.append(final_prompt)
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        # Accumulate results
        all_prompts.extend(prompts)
        all_preds.extend(preds)
        all_labels.extend(labels)
        gc.collect()

    # Write all results at once outside the loop
    with open(save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            result = {
                "prompt": text, 
                "predict": pred, 
                "label": label,
                "few_shot_enabled": enable_few_shot,
                "n_shot": n_shot if enable_few_shot else 0,
                "few_shot_format": few_shot_format if enable_few_shot else None
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(all_prompts)} total generated results have been saved at {save_name}.")
    if enable_few_shot:
        print(f"🎯 few-shot settings: enabled={enable_few_shot}, number of examples={len(few_shot_examples)}, format={few_shot_format}")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
