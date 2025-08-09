#!/usr/bin/env python3
"""
    inference and evaluation pipeline class
    contains test data preprocessing, inference command generation and Hugging Face evaluation functionality
"""

import os
import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import subprocess
import time

# config logger
logger = logging.getLogger(__name__)

# common evaluation metrics constants
# COMMON_QA_METRICS = ['exact_match', 'f1']


class ClassificationEvaluator:
    """
    inference and evaluation pipeline class
    
    features:
    1. inference: check and supplement test json file content, return inference command
    2. evaluation: use Hugging Face evaluate to calculate exact-match and F1 metrics
    """
    
    def __init__(self, 
                 workspace_dir: str,
                 llamafactory_dir: str,
                 default_system_prompt: str = "You are a helpful assistant."):
        """
        initialize inference evaluation pipeline
        
        Args:
            workspace_dir: workspace directory
            llamafactory_dir: LLaMA-Factory directory
            default_system_prompt: default system prompt
        """
        self.workspace_dir = Path(workspace_dir)
        self.llamafactory_dir = Path(llamafactory_dir)
        self.default_system_prompt = default_system_prompt
        
    
    def load_template(self, template_file: str) -> Optional[Dict[str, str]]:
        """
        load template file (reference transfer2instruction_tuning.py implementation)
        
        Args:
            template_file: template file path (.py file)
            
        Returns:
            template dictionary or None
        """
        if not os.path.exists(template_file):
            logger.error(f"template file not found: {template_file}")
            return None
        
        # dynamic import template file
        sys.path.insert(0, os.path.dirname(os.path.abspath(template_file)))
        module_name = os.path.splitext(os.path.basename(template_file))[0]
        
        try:
            template_module = __import__(module_name)
            if hasattr(template_module, 'TEMPLATE'):
                logger.info(f"✅ template loaded successfully: {template_file}")
                return template_module.TEMPLATE
            else:
                logger.error(f"template file not found TEMPLATE variable")
                return None
        except Exception as e:
            logger.error(f"failed to load template file: {e}")
            return None
    
    def get_nested_value(self, obj: Dict[str, Any], path: str) -> str:
        """
        get nested object value, support dot separated path
        
        Args:
            obj: data object
            path: path string, e.g. "Question" or "Answer.Value"
            
        Returns:
            string value
        """
        try:
            keys = path.split('.')
            value = obj
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, list) and key.isdigit():
                    value = value[int(key)]
                else:
                    return ""
            
            # if complex object, convert to string
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False)
            
            return str(value) if value is not None else ""
        except:
            return ""
    
    def replace_template_vars(self, template_str: str, sample: Dict[str, Any]) -> str:
        """
        replace variables in template string (reference transfer2instruction_tuning.py)
        
        Args:
            template_str: template string, e.g. "answer the question: {sample.Question}"
            sample: sample data
            
        Returns:
            replaced string
        """
        def replace_match(match):
            var_path = match.group(1)  # get sample.xxx.yyy part
            if var_path.startswith('sample.'):
                field_path = var_path[7:]  # remove 'sample.' prefix
                return self.get_nested_value(sample, field_path)
            return match.group(0)  # if not start with sample., keep original
        
        # find all {sample.xxx} pattern
        pattern = r'\{(sample\.[^}]+)\}'
        result = re.sub(pattern, replace_match, template_str)
        return result
    
    def apply_template(self, template: Dict[str, str], sample_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        apply template to sample data
        
        Args:
            template: template dictionary
            sample_data: sample data
            
        Returns:
            converted data or None
        """
        try:
            result = {}
            for key, template_str in template.items():
                if isinstance(template_str, str):
                    result[key] = self.replace_template_vars(template_str, sample_data)
                else:
                    result[key] = str(template_str)
            
            return result
        except Exception as e:
            logger.error(f"failed to apply template: {e}")
            return None
    
    def check_and_transform_with_template(self, test_json_file: str, template_file: Optional[str] = None) -> str:
        """
        use template to check and convert test json file content
        
        Args:
            test_json_file: test data json file path
            template_file: template file path (.py file), if None, use default supplement logic
            
        Returns:
            converted json file path
        """
        logger.info(f"🔍 check and convert test data file: {test_json_file}")
        
        try:
            # read original data
            with open(test_json_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # ensure data is list format
            if isinstance(test_data, dict):
                if 'Data' in test_data:
                    test_data = test_data['Data']
                else:
                    test_data = [test_data]
            
            logger.info(f"📊 original data: {len(test_data)} samples")
            
            # if template file is provided, use template to convert
            if template_file:
                template = self.load_template(template_file)
                if template:
                    return self._transform_with_template(test_data, template, test_json_file)
            
            # otherwise use default supplement logic
            return self._supplement_with_defaults(test_data, test_json_file)
            
        except Exception as e:
            logger.error(f"❌ data conversion failed: {e}")
            return test_json_file
    
    def _transform_with_template(self, test_data: List[Dict], template: Dict[str, str], original_file: str) -> str:
        """use template to convert data"""
        logger.info("🔄 use template to convert data...")
        
        converted_data = []
        for i, item in enumerate(test_data):
            try:
                result = self.apply_template(template, item)
                if result:
                    # ensure id field
                    if 'id' not in result:
                        result['id'] = f"sample_{i}"
                    converted_data.append(result)
                else:
                    logger.warning(f"skip sample {i}: template application failed")
            except Exception as e:
                logger.error(f"error processing sample {i}: {e}")
        
        # save converted data
        original_path = Path(original_file)
        converted_file = original_path.parent / f"{original_path.stem}_template_converted{original_path.suffix}"
        
        with open(converted_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ template conversion completed: {len(converted_data)} valid samples")
        logger.info(f"   - converted file: {converted_file}")
        
        return str(converted_file)
    
    def _supplement_with_defaults(self, test_data: List[Dict], original_file: str) -> str:
        """use default logic to supplement data"""
        logger.info("🔄 use default logic to supplement data...")
        
        modifications_count = 0
        
        for i, item in enumerate(test_data):
            if not isinstance(item, dict):
                continue
            
            # check and supplement fields
            if 'system' not in item or not item['system']:
                item['system'] = self.default_system_prompt
                modifications_count += 1
            
            if 'instruction' not in item or not item['instruction']:
                item['instruction'] = "Please answer the following question."
                modifications_count += 1
            
            if 'input' not in item:
                item['input'] = ""
            
            if 'id' not in item:
                item['id'] = f"sample_{i}"
        
        # save supplemented data
        original_path = Path(original_file)
        supplemented_file = original_path.parent / f"{original_path.stem}_supplemented{original_path.suffix}"
        
        with open(supplemented_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ default supplement completed: {len(test_data)} samples, {modifications_count} modifications")
        logger.info(f"   - supplemented file: {supplemented_file}")
        
        return str(supplemented_file)
    
    def _resolve_test_file_path(self, test_input: str) -> str:
        """
        parse test file path: support direct file path or find through dataset_info.json
        
        Args:
            test_input: file path or dataset name
            
        Returns:
            actual file path
        """
        # if absolute path or relative path and file exists, return directly
        if os.path.exists(test_input):
            logger.info(f"✅ use direct file path: {test_input}")
            return test_input
        
        # try to find through dataset_info.json
        dataset_info_path = self.llamafactory_dir / "data" / "dataset_info.json"
        
        if not dataset_info_path.exists():
            logger.warning(f"⚠️ dataset_info.json not found: {dataset_info_path}")
            return test_input
        
        try:
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            
            # check if it is a dataset name
            if test_input in dataset_info:
                dataset_config = dataset_info[test_input]
                
                # get file name
                if 'file_name' in dataset_config:
                    file_name = dataset_config['file_name']
                    # build full path
                    full_path = self.llamafactory_dir / "data" / file_name
                    
                    if full_path.exists():
                        logger.info(f"✅ find file from dataset_info: {test_input} -> {full_path}")
                        return str(full_path)
                    else:
                        logger.warning(f"⚠️ file not found in dataset_info: {full_path}")
                else:
                    logger.warning(f"⚠️ dataset {test_input} config has no file_name field")
            else:
                logger.info(f"📝 {test_input} not in dataset_info, try as file path")
        
        except Exception as e:
            logger.error(f"❌ parse dataset_info.json failed: {e}")
        
        # if all failed, try to find in data directory directly
        data_dir_path = self.llamafactory_dir / "data" / test_input
        if data_dir_path.exists():
            logger.info(f"✅ find file in data directory: {data_dir_path}")
            return str(data_dir_path)
        
        # finally return original input (maybe relative path)
        logger.warning(f"⚠️ failed to parse file path, use original input: {test_input}")
        return test_input
    

    def inference(self, 
                  model_path: str, 
                  test_name: str, 
                  train_name: str,
                  model_config: Dict[str, Any],
                  output_file: str,
                  template_file: Optional[str] = None) -> Tuple[str, str]:
        """
        inference function: check and supplement test data and generate inference command
        
        Args:
            model_path: model path
            test_json_file: test data json file path
            model_config: model config dictionary
            template_file: template file path (.py file), if provided, use template to convert data format
            
        Returns:
            Tuple[inference command, output file path]
        """
        logger.info(f"🚀 start inference preparation: {model_path}")
        
        try:
            # 1. use template to check and convert test data format
            # test_json_file to find in dataset_info.json
            test_json_file = self._resolve_test_file_path(test_name)
            # processed_test_file = self.check_and_transform_with_template(test_json_file, template_file)
            
            # 2. determine output file path
            # model_name = 'baseline' if baseline else model_config.get('name', 'model')
            # output_file = self.inference_dir / f"{test_name}_{model_name}_inference.jsonl"
            
            # 3. build inference parameters
            inference_args = self._build_inference_args(
                model_path, test_name, train_name, model_config, str(output_file)
            )
            
            # 4. build full command
            inference_cmd = f"python scripts/vllm_infer.py {' '.join(inference_args)}"
            full_cmd = f"cd {self.llamafactory_dir} && {inference_cmd}"
            
            logger.info(f"📝 inference command generated:")
            logger.info(f"   - command: {full_cmd}")
            
            return full_cmd, str(output_file)
            
        except Exception as e:
            logger.error(f"❌ inference command generation failed: {e}")
            raise
    
    def _build_inference_args(self, 
                             model_path: str, 
                             test_file: str, 
                             train_file:str,
                             model_config: Dict[str, Any], 
                             output_file: str) -> List[str]:
        """
        build inference parameter list
        
        Args:
            model_path: model path
            test_file: test file path
            model_config: model config
            output_file: output file path
            
        Returns:
            inference parameter list
        """
        # basic inference parameters
        few_shot = model_config.get('few_shot',False)
        if few_shot:
            if model_config.get('finetuning_type') == 'lora':
                # LoRA model needs to specify base model and adapter separately
                inference_args = [
                    f"--model_name_or_path {model_config['base_model']}",
                    f"--adapter_name_or_path {model_path}",
                    f"--dataset {test_file}",
                    f"--train_dataset {train_file}",
                    f"--n_shot {model_config['n_shot']}",
                    f"--save_name {output_file}"
                ]
            else:
                # full-finetuning model directly uses the trained model path
                inference_args = [
                    f"--model_name_or_path {model_path}",
                    f"--dataset {test_file}",
                    f"--train_dataset {train_file}",
                    f"--n_shot {model_config['n_shot']}",
                    f"--save_name {output_file}"
                ]
        else:
            if model_config.get('finetuning_type') == 'lora':
            # LoRA model needs to specify base model and adapter separately
                inference_args = [
                    f"--model_name_or_path {model_config['base_model']}",
                    f"--adapter_name_or_path {model_path}",
                    f"--dataset {test_file}",
                    f"--save_name {output_file}"
                ]
            else:
                # full-finetuning model directly uses the trained model path
                inference_args = [
                    f"--model_name_or_path {model_path}",
                    f"--dataset {test_file}",
                    f"--save_name {output_file}"
                ]
        
        # add optional parameters
        optional_params = {
            'template': model_config.get('template'),
            'cutoff_len': model_config.get('cutoff_len', 4096),
            'batch_size': model_config.get('inference_batch_size', 64),
            'max_samples': model_config.get('max_samples'),
            'temperature': model_config.get('temperature', 0.1),
            'top_p': model_config.get('top_p', 0.9),
            'max_new_tokens': model_config.get('max_new_tokens', 512)
        }
        
        for param, value in optional_params.items():
            if value is not None:
                inference_args.append(f"--{param} {value}")
        
        return inference_args
    
    def evaluation(self, 
                   prediction_file: str, 
                   ground_truth_file: str,
                   metrics: List[str] = ['accuracy'],
                   output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        evaluation function: use Hugging Face evaluate to calculate specified metrics
        
        Args:
            prediction_file: prediction result file path (JSONL format)
            ground_truth_file: ground truth data register name
            metrics: list of metrics to calculate, e.g. ['exact_match', 'f1', 'bleu', 'rouge']
            output_dir: result output directory
            
        Returns:
            evaluation result dictionary
        """
        # set default metrics
        
        logger.info(f"📊 start evaluation: {prediction_file}")
        logger.info(f"📏 specified metrics: {metrics}")
        ground_truth_path = self._resolve_test_file_path(ground_truth_file)
        try:
            # 1. load prediction results and ground truth
            predictions, references = self._load_prediction_and_ground_truth(
                prediction_file, ground_truth_path
            )
            
            logger.info(f"📋 data statistics:")
            logger.info(f"   - prediction samples: {len(predictions)}")
            logger.info(f"   - ground truth samples: {len(references)}")
            
            # 2. use Hugging Face evaluate to calculate specified metrics
            eval_results = self._compute_metrics_with_evaluate(predictions, references, metrics)
            
            # 3. add extra statistics
            eval_results.update({
                'prediction_file': prediction_file,
                'metrics_used': metrics,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # 4. save results
            if output_dir:
                self._save_evaluation_results(eval_results, output_dir, prediction_file)
            
            # 5. print result summary
            logger.info(f"✅ evaluation completed:")
            for metric in metrics:
                if metric in eval_results:
                    value = eval_results[metric]
                    if isinstance(value, (int, float)):
                        logger.info(f"   - {metric}: {value:.4f}")
                    else:
                        logger.info(f"   - {metric}: {value}")
            
            return eval_results
            
        except Exception as e:
            logger.error(f"❌ evaluation failed: {e}")
            raise
    
    def _load_prediction_and_ground_truth(self, 
                                         prediction_file: str, 
                                         ground_truth_file: str) -> Tuple[List[str], List[str]]:
        """
        load prediction results and ground truth
        
        Args:
            prediction_file: prediction file path
            ground_truth_file: ground truth file path
            
        Returns:
            Tuple[prediction list, ground truth list]
        """
        # load prediction results (JSONL format)
        predictions = []
        with open(prediction_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # support multiple prediction result formats
                    pred = data.get('predict', data.get('prediction', data.get('output', '')))
                    predictions.append(str(pred).strip())
        
        # load ground truth
        references = []
        if ground_truth_file.endswith('.jsonl'):
            # JSONL format
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        ref = data.get('output', data.get('answer', data.get('ground_truth', '')))
                        references.append(str(ref).strip())
        else:
            # JSON format
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        ref = item.get('output', item.get('answer', item.get('ground_truth', '')))
                        references.append(str(ref).strip())
                else:
                    # single sample
                    ref = data.get('output', data.get('answer', data.get('ground_truth', '')))
                    references.append(str(ref).strip())
        
        # ensure prediction and label number consistent
        min_len = min(len(predictions), len(references))
        if len(predictions) != len(references):
            logger.warning(f"prediction number({len(predictions)}) and label number({len(references)}) inconsistent, use first {min_len} samples")
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        return predictions, references
    
    def _accuracy_score(self, prediction: str, ground_truth: str) -> float:
        """
        calculate accuracy score, avoid misjudgment by matching at the end
        
        Args:
            prediction: prediction result (model output)
            ground_truth: ground truth
            
        Returns:
            accuracy score (1.0 means correct, 0.0 means incorrect)
        """
        pred = prediction.strip()
        gt = ground_truth.strip()
        
        # exact match
        if pred.lower() == gt.lower():
            return 1.0
        
        # clean prediction end punctuation
        pred_clean = self._clean_end_punctuation(pred.lower())
        gt_lower = gt.lower()
        
        # 1. check if the prediction ends with the ground truth
        if pred_clean.endswith(gt_lower):
            return 1.0
        
        # 2. check if the prediction ends with the ground truth (single letter for multiple choice questions)
        if len(gt) == 1 and gt.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return self._match_choice_at_end(pred_clean, gt_lower)
        
        # 3. check if the prediction ends with the ground truth (number answer)
        if gt.isdigit():
            return self._match_number_at_end(pred_clean, gt)
        
        # 4. check if the prediction ends with the ground truth (last few words)
        return self._match_last_words(pred_clean, gt_lower)

    def _clean_end_punctuation(self, text: str) -> str:
        """remove end punctuation"""
        punctuation = '.。,，!！?？;；:：、)）]】}}'
        while text and text[-1] in punctuation:
            text = text[:-1]
        return text.strip()

    def _match_choice_at_end(self, pred_clean: str, gt_lower: str) -> float:
        """check if the last character of the prediction is the ground truth (multiple choice questions)"""
        if not pred_clean:
            return 0.0
        
        # check if the last character of the prediction is the ground truth
        if pred_clean[-1] == gt_lower:
            return 1.0
        
        # check if the last word of the prediction is the ground truth
        words = pred_clean.split()
        if words and words[-1] == gt_lower:
            return 1.0
        
        return 0.0

    def _match_number_at_end(self, pred_clean: str, gt: str) -> float:
        """check if the last number of the prediction is the ground truth"""
        if 'true' in pred_clean.lower():
            pred_clean = '1'
        elif 'false' in pred_clean.lower():
            pred_clean = '0'
        words = pred_clean.split()
        if not words:
            return 0.0
        
        # check if the last word of the prediction is the ground truth
        if words[-1] == gt:
            return 1.0
        
        # check if the last word of the prediction contains the ground truth (handle cases like "answer123")
        if words[-1].endswith(gt) and words[-1][-len(gt):] == gt:
            # ensure the previous character is not a number (avoid cases like "1234" matching "34")
            if len(words[-1]) == len(gt) or not words[-1][-len(gt)-1].isdigit():
                return 1.0
        
        return 0.0

    def _match_last_words(self, pred_clean: str, gt_lower: str) -> float:
        """check if the last few words of the prediction is the ground truth"""
        pred_words = pred_clean.split()
        gt_words = gt_lower.split()
        
        if not pred_words or not gt_words:
            return 0.0
        
        # if the ground truth is multiple words, check the last word group
        if len(gt_words) > 1:
            if len(pred_words) >= len(gt_words):
                pred_suffix = ' '.join(pred_words[-len(gt_words):])
                if pred_suffix == gt_lower:
                    return 1.0
        
        # if the ground truth is a single word, check the last word
        elif len(gt_words) == 1:
            if pred_words[-1] == gt_words[0]:
                return 1.0
        
        return 0.0

    def _compute_metrics_with_evaluate(self, predictions: List[str], references: List[str], metrics: List[str]) -> Dict[str, Any]:
        """
        use Hugging Face evaluate to calculate specified metrics
        
        Args:
            predictions: prediction result list
            references: ground truth list
            metrics: list of metrics to calculate
            
        Returns:
            evaluation metrics dictionary
        """
        try:
            import evaluate
            logger.info("✅ use Hugging Face evaluate to calculate metrics")
            
            results = {}
            
            for metric_name in metrics:
                try:
                    logger.info(f"🔄 calculate metric: {metric_name}")
                    
                    if metric_name in ['accuracy']:
                        # use custom accuracy calculation method
                        accuracy_scores = [self._accuracy_score(pred, ref) for pred, ref in zip(predictions, references)]
                        results[metric_name] = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
                        logger.info(f"✅ {metric_name} calculation completed: {results[metric_name]:.4f}")
                    else:
                        # other metrics use Hugging Face evaluate
                        try:
                            metric = evaluate.load('metrics/'+metric_name)
                        except Exception as e:
                            logger.error(f'invalid metric: {e}')
                        metric_result = metric.compute(predictions=predictions, references=references)
                        results[metric_name] = metric_result
                        logger.info(f"✅ {metric_name} calculation completed")
                    
                except Exception as e:
                    logger.error(f"❌ metric {metric_name} load or calculate failed: {e}")
                    results[metric_name] = None
            
            return results
            
        except ImportError:
            logger.error("❌ evaluate library not installed, cannot calculate metrics")
            raise ImportError("please install evaluate library: pip install evaluate")
        except Exception as e:
            logger.error(f"❌ evaluate library error: {e}")
            raise
      
    
    def _save_evaluation_results(self, 
                                results: Dict[str, Any], 
                                output_dir: str, 
                                prediction_file: str):
        """
        save evaluation results
        
        Args:
            results: evaluation results
            output_dir: output directory
            prediction_file: prediction file path (used to generate file name)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # generate result file name
        pred_name = Path(prediction_file).stem
        result_file = output_path / f"{pred_name}_evaluation_results.json"
        
        # save results
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 evaluation results saved: {result_file}")
    



# example usage function
def example_usage():
    """example usage method"""
    
    # initialize pipeline
    pipeline = ClassificationEvaluator(
        workspace_dir="/path/to/workspace",
        llamafactory_dir="/path/to/LLaMA-Factory"
    )
    
    # model configuration example
    model_config = {
        'name': 'llama3_sft',
        'base_model': 'meta-llama/Llama-3-8B-Instruct',
        'finetuning_type': 'full',  # or 'lora'
        'inference_batch_size': 64,
        'temperature': 0.1,
        'max_new_tokens': 512
    }
    
    # 1. use template to convert data and generate inference command
    cmd, output_file = pipeline.inference(
        model_path="/path/to/trained/model",
        test_json_file="/path/to/test.json",
        model_config=model_config,
        template_file="evaluation/qa_template_example.py"  # use template to convert
    )
    print(f"inference command: {cmd}")
    
    # 2. no template, only default supplement (backward compatibility)
    cmd, output_file = pipeline.inference(
        model_path="/path/to/trained/model",
        test_json_file="/path/to/test.json",
        model_config=model_config,
        template_file=None  # use default supplement logic
    )
    print(f"inference command (default supplement): {cmd}")
    
    # 3. execute evaluation - use default metrics (exact_match, f1)
    eval_results = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json"
    )
    print(f"evaluation results (default metrics): {eval_results}")
    
    # 4. execute evaluation - specify multiple metrics
    eval_results_multi = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        metrics=['exact_match', 'f1', 'bleu', 'rouge', 'meteor', 'accuracy']
    )
    print(f"evaluation results (multiple metrics): {eval_results_multi}")
    
    # 5. execute evaluation - only use specific metrics
    eval_results_specific = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        metrics=['bertscore', 'rouge1', 'rouge2', 'rougeL','accuracy']
    )
    print(f"evaluation results (specific metrics): {eval_results_specific}")



if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # run example
    example_usage()