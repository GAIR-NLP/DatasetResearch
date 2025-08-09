# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
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
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template


if TYPE_CHECKING:
    from numpy.typing import NDArray


class Evaluator:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]

    @torch.inference_mode()
    def batch_inference(self, batch_input: dict[str, "torch.Tensor"]) -> tuple[list[str], list[float], list[list[float]]]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        
        predictions = [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]
        confidences = torch.max(choice_probs, dim=-1)[0].float().cpu().numpy().tolist()  # 最大概率作为置信度
        probabilities = choice_probs.float().cpu().numpy().tolist()  # 完整概率分布
        
        return predictions, confidences, probabilities

    def eval(self) -> None:
        eval_task = self.eval_args.task.split("_")[0]
        eval_split = self.eval_args.task.split("_")[1]

        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, encoding="utf-8") as f:
            categorys: dict[str, dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        all_confidences = []  # 存储所有置信度
        all_probabilities = []  # 存储所有概率分布
        all_predictions = []  # 存储所有预测
        all_labels = []  # 存储所有标签
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            subject_confidences, subject_probabilities = [], []
            for i in trange(len(dataset[eval_split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[eval_split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds, confs, probs = self.batch_inference(batch_input)
                outputs += preds
                subject_confidences += confs
                subject_probabilities += probs

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            
            # 存储详细结果，包含置信度、概率和ground truth
            results[subject] = {
                "predictions": {str(i): outputs[i] for i in range(len(outputs))},
                "confidences": subject_confidences,
                "probabilities": subject_probabilities,
                "labels": labels,  # ground truth labels
                "ground_truth": labels,  # 明确标记为ground truth
                "accuracy": float(np.mean(corrects))
            }
            
            # 累积全局数据
            all_confidences.extend(subject_confidences)
            all_probabilities.extend(subject_probabilities)
            all_predictions.extend(outputs)
            all_labels.extend(labels)

        pbar.close()
        
        # 创建综合结果
        comprehensive_results = {
            "category_corrects": category_corrects,
            "detailed_results": results,
            "global_data": {
                "confidences": all_confidences,
                "probabilities": all_probabilities, 
                "predictions": all_predictions,
                "labels": all_labels,
                "ground_truth": all_labels,  # 明确标记为ground truth
                "total_samples": len(all_predictions),
                "correct_predictions": [pred == label for pred, label in zip(all_predictions, all_labels)]
            }
        }
        
        self._save_results(comprehensive_results)

    def _save_results(self, comprehensive_results: dict) -> None:
        category_corrects = comprehensive_results["category_corrects"]
        detailed_results = comprehensive_results["detailed_results"]
        global_data = comprehensive_results["global_data"]
        
        # 计算基础分类准确率
        score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)  # 改为exist_ok=True
            
            # 保存详细结果（包含置信度和概率）
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(detailed_results, f, indent=2)
            
            # 保存全局数据（用于计算ECE等指标）
            with open(os.path.join(self.eval_args.save_dir, "global_data.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(global_data, f, indent=2)
            
            # 保存预测与ground truth对比文件
            comparison_data = []
            for i, (pred, gt, conf, prob) in enumerate(zip(
                global_data["predictions"], 
                global_data["ground_truth"], 
                global_data["confidences"],
                global_data["probabilities"]
            )):
                comparison_data.append({
                    "sample_id": i,
                    "prediction": pred,
                    "ground_truth": gt,
                    "correct": pred == gt,
                    "confidence": conf,
                    "probabilities": {
                        "A": prob[0],
                        "B": prob[1], 
                        "C": prob[2],
                        "D": prob[3]
                    },
                    "predicted_prob": prob[ord(pred) - ord("A")],
                    "ground_truth_prob": prob[ord(gt) - ord("A")]
                })
            
            with open(os.path.join(self.eval_args.save_dir, "predictions_vs_ground_truth.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(comparison_data, f, indent=2)
            
            # 保存基础metrics（兼容性）
            basic_metrics = {}
            for category_name, category_correct in category_corrects.items():
                if len(category_correct) > 0:
                    basic_metrics[category_name] = float(np.mean(category_correct))
            
            with open(os.path.join(self.eval_args.save_dir, "metrics.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(basic_metrics, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()
