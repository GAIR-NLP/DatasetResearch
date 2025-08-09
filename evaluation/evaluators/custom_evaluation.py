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
import evaluate
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import subprocess
import time

# configure logging
logger = logging.getLogger(__name__)

# common evaluation metrics constants
# COMMON_QA_METRICS = ['exact_match', 'f1']


class QuestionAnsweringEvaluator:
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
        
        # set directory structure
        self.inference_dir = self.llamafactory_dir / "inference_results"
        self.eval_results_dir = self.workspace_dir / "evaluation" / "results"
        
        # create necessary directories
        for dir_path in [self.inference_dir, self.eval_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
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
        
        # dynamically import template file
        sys.path.insert(0, os.path.dirname(os.path.abspath(template_file)))
        module_name = os.path.splitext(os.path.basename(template_file))[0]
        
        try:
            template_module = __import__(module_name)
            if hasattr(template_module, 'TEMPLATE'):
                logger.info(f"✅ template loaded successfully: {template_file}")
                return template_module.TEMPLATE
            else:
                logger.error(f"TEMPLATE variable not found in template file")
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
            
            # if it is a complex object, convert to string
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
        
        # find all {sample.xxx} patterns
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
        use template to check and convert test JSON file content
        
        Args:
            test_json_file: test data JSON file path
            template_file: template file path (.py file), if None, use default supplement logic
            
        Returns:
            converted JSON file path
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
            
            # if no template file, use default supplement logic
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
        resolve test file path: support direct file path or find through dataset_info.json
        
        Args:
            test_input: file path or dataset name
            
        Returns:
            actual file path
        """
        # if it is an absolute path or relative path and the file exists, return directly
        if os.path.exists(test_input):
            logger.info(f"✅ use file path directly: {test_input}")
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
                    logger.warning(f"⚠️ dataset {test_input} configuration has no file_name field")
            else:
                logger.info(f"📝 {test_input} not in dataset_info, try as file path")
        
        except Exception as e:
            logger.error(f"❌ failed to parse dataset_info.json: {e}")
        
        # if all above fail, try to find in data directory directly
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
                  baseline: bool = False,
                  template_file: Optional[str] = None) -> Tuple[str, str]:
        """
        inference function: check and supplement test data and generate inference command
        
        Args:
            model_path: model path
            test_json_file: test data JSON file path
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
            model_name = 'baseline' if baseline else model_config.get('name', 'model')
            output_file = self.inference_dir / f"{test_name}_{model_name}_inference.jsonl"
            
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
            logger.error(f"❌ failed to generate inference command: {e}")
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
                    f"--save_name {output_file}"
                ]
            else:
                # full-finetuned model directly uses the trained model path
                inference_args = [
                    f"--model_name_or_path {model_path}",
                    f"--dataset {test_file}",
                    f"--train_dataset {train_file}",
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
                # full-finetuned model directly uses the trained model path
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
                   metrics: List[str] = ['exact_match', 'f1'],
                   output_dir: Optional[str] = None,
                   evaluation_script: Optional[str] = None,
                   script_args: Optional[Dict[str, Any]] = None,
                   save_detailed: bool = False) -> Dict[str, Any]:
        """
        evaluation function: support custom script and HF metrics calculation
        
        Args:
            prediction_file: prediction result file path (JSONL format)
            ground_truth_file: ground truth file path (JSON/JSONL format)
            metrics: list of metrics to calculate, e.g. ['exact_match', 'f1', 'bleu', 'rouge']
            output_dir: result output directory
            evaluation_script: custom evaluation script path (relative to LLaMA-Factory/evaluation/)
            script_args: additional parameters passed to the evaluation script
            save_detailed: whether to save detailed results (note: significantly increases memory and storage usage)
            
        Returns:
            evaluation result dictionary
        """
        logger.info(f"📊 start evaluation: {prediction_file}")
        ground_truth_path = self._resolve_test_file_path(ground_truth_file)
        
        try:
            # 1. if evaluation script is specified, use custom script
            if evaluation_script:
                if not script_args:
                    script_args = {}
                script_args['save_detailed'] = save_detailed  # pass detailed result save option
                
                eval_results = self._evaluate_with_script(
                    prediction_file, ground_truth_path, evaluation_script, script_args
                )
            else:
                # otherwise use default Hugging Face evaluate method
                logger.info(f"📏 use default metrics: {metrics}")
                eval_results = self._evaluate_with_hf_metrics(
                    prediction_file, ground_truth_path, metrics
                )
            
            # add additional statistics
            eval_results.update({
                'prediction_file': prediction_file,
                'evaluation_method': evaluation_script if evaluation_script else 'huggingface_metrics',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # save results
            if output_dir:
                self._save_evaluation_results(eval_results, output_dir, prediction_file)
            
            # print result summary
            self._print_evaluation_summary(eval_results)
            
            return eval_results
            
        except Exception as e:
            logger.error(f"❌ evaluation failed: {e}")
            raise
    
    def _evaluate_with_script(self, 
                                 prediction_file: str, 
                                 ground_truth_file: str,
                                 evaluation_script: str,
                                 script_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        use custom evaluation script for evaluation (support new unified interface)
        
        Args:
            prediction_file: prediction file path
            ground_truth_file: ground truth file path
            evaluation_script: evaluation script path (relative to LLaMA-Factory/evaluation/)
            script_args: additional parameters
            
        Returns:
            evaluation result dictionary
        """
        # first try to use v2 version (unified interface)
        script_name = os.path.basename(evaluation_script)
        script_dir = os.path.dirname(evaluation_script)
        script_base = os.path.splitext(script_name)[0]
        v2_script = f"{script_dir}/{script_base}_v2.py"
        
        # check if v2 version exists
        v2_script_path = self.llamafactory_dir / "evaluation" / v2_script
        original_script_path = self.llamafactory_dir / "evaluation" / evaluation_script
        
        if v2_script_path.exists():
            logger.info(f"🔧 use unified interface script: {v2_script}")
            selected_script = v2_script_path
            is_unified = True
        elif original_script_path.exists():
            logger.info(f"🔧 use original script: {evaluation_script}")
            selected_script = original_script_path
            is_unified = False
        else:
            raise FileNotFoundError(f"evaluation script not found: {original_script_path} and {v2_script_path}")
        
        # build command line arguments
        cmd_args = [
            "python", str(selected_script),
            "--prediction_file", prediction_file,
            "--dataset_file", ground_truth_file
        ]
        
        # add output file parameter (temporary file)
        import tempfile
        temp_output_file = None
        
        if is_unified:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_output_file = f.name
            cmd_args.extend(["--output_file", temp_output_file])
        
        # add additional parameters
        if script_args:
            for key, value in script_args.items():
                if value is not None:
                    if isinstance(value, bool):
                        if value:  # only add flag when True
                            cmd_args.append(f"--{key}")
                    else:
                        cmd_args.extend([f"--{key}", str(value)])
        
        logger.info(f"🚀 execute evaluation script: {' '.join(cmd_args)}")
        
        try:
            # execute evaluation script
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=str(self.llamafactory_dir),
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"❌ evaluation script execution failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"evaluation script execution failed, exit code: {result.returncode}")
            
            # parse output results
            if is_unified and temp_output_file and os.path.exists(temp_output_file):
                # read structured results from JSON file
                try:
                    with open(temp_output_file, 'r', encoding='utf-8') as f:
                        eval_results = json.load(f)
                    
                    logger.info(f"✅ successfully parsed unified interface results")
                    logger.info(f"📊 metrics: {list(eval_results.get('metrics', {}).keys())}")
                    
                    return eval_results
                    
                except Exception as e:
                    logger.error(f"❌ failed to parse JSON results: {e}")
                    # fallback to parse stdout
                    eval_results = self._parse_script_output(result.stdout, evaluation_script)
            else:
                # parse stdout output (original script)
                eval_results = self._parse_script_output(result.stdout, evaluation_script)
            
            logger.info(f"✅ specialized evaluation script executed successfully")
            return eval_results
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ evaluation script execution timeout")
            raise RuntimeError("evaluation script execution timeout")
        except Exception as e:
            logger.error(f"❌ error executing evaluation script: {e}")
            raise
        finally:
            # ensure temporary file is cleaned up
            if temp_output_file and os.path.exists(temp_output_file):
                try:
                    os.remove(temp_output_file)
                except:
                    pass
            raise
    
    def _parse_script_output(self, output: str, script_type: str) -> Dict[str, Any]:
        """
        parse evaluation script output results
        
        Args:
            output: script stdout output
            script_type: script type for determining parsing method
            
        Returns:
            evaluation results after parsing
        """
        results = {}
        
        if "triviaqa" in script_type.lower():
            # parse TriviaQA script output
            lines = output.strip().split('\n')
            for line in lines:
                if "Exact Match:" in line:
                    try:
                        results['exact_match'] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "F1 Score:" in line:
                    try:
                        results['f1'] = float(line.split(":")[1].strip())
                    except:
                        pass
                elif "Total Questions:" in line:
                    try:
                        results['total_samples'] = int(line.split(":")[1].strip())
                    except:
                        pass
                elif "Successfully Evaluated:" in line:
                    try:
                        results['evaluated_samples'] = int(line.split(":")[1].strip())
                    except:
                        pass
        
        elif "bigcodebench" in script_type.lower():
            # parse BigCodeBench script output
            lines = output.strip().split('\n')
            for line in lines:
                if "Passed:" in line:
                    try:
                        results['passed'] = int(re.search(r'(\d+)', line).group(1))
                    except:
                        pass
                elif "Total:" in line:
                    try:
                        results['total_samples'] = int(re.search(r'(\d+)', line).group(1))
                    except:
                        pass
                elif "Accuracy:" in line:
                    try:
                        results['accuracy'] = float(re.search(r'([\d.]+)', line).group(1))
                    except:
                        pass
                elif "pass@" in line.lower():
                    try:
                        # parse pass@k metric
                        match = re.search(r'pass@(\d+)[:\s]+([\d.]+)', line)
                        if match:
                            k, value = match.groups()
                            results[f'pass@{k}'] = float(value)
                    except:
                        pass
        
        else:
            # generic parsing: try to identify common metric formats
            lines = output.strip().split('\n')
            for line in lines:
                # try to match "Metric: Value" format
                match = re.search(r'([A-Za-z_][A-Za-z0-9_\s]*)[:\s]+([\d.]+)', line)
                if match:
                    metric_name = match.group(1).strip().lower().replace(' ', '_')
                    value = float(match.group(2))
                    results[metric_name] = value
        
        # if no results are parsed, save original output
        if not results:
            results['raw_output'] = output
            logger.warning("⚠️ cannot parse evaluation script output, save original output")
        
        return results
    
    def _evaluate_with_hf_metrics(self, 
                                 prediction_file: str, 
                                 ground_truth_file: str,
                                 metrics: List[str]) -> Dict[str, Any]:
        """
        use Hugging Face evaluate to calculate metrics (original evaluation method)
        
        Args:
            prediction_file: prediction file path
            ground_truth_file: ground truth file path
            metrics: list of metrics to calculate
            
        Returns:
            evaluation result dictionary
        """
        # 1. load prediction results and ground truth
        predictions, references = self._load_prediction_and_ground_truth(
            prediction_file, ground_truth_file
            )
            
        logger.info(f"📋 data statistics:")
        logger.info(f"   - prediction samples: {len(predictions)}")
        logger.info(f"   - ground truth samples: {len(references)}")
        
        # 2. use Hugging Face evaluate to calculate specified metrics
        eval_results = self._compute_metrics_with_evaluate(predictions, references, metrics)
        
        # 3. add statistics
        eval_results.update({
            'total_samples': len(predictions),
        'metrics_used': metrics
    })
        
        return eval_results
    
    def _print_evaluation_summary(self, eval_results: Dict[str, Any]):
        """
        打印评估结果摘要
        
        Args:
            eval_results: evaluation result dictionary
        """
        logger.info(f"✅ evaluation completed:")
        
        # print common metrics
        common_metrics = ['exact_match', 'f1', 'accuracy', 'passed', 'total_samples']
        for metric in common_metrics:
                if metric in eval_results:
                    value = eval_results[metric]
                    if isinstance(value, (int, float)):
                        if metric in ['exact_match', 'f1', 'accuracy']:
                            logger.info(f"   - {metric}: {value:.4f}")
                        else:
                            logger.info(f"   - {metric}: {value}")
            
        # print pass@k metrics
        for key, value in eval_results.items():
            if key.startswith('pass@') and isinstance(value, (int, float)):
                logger.info(f"   - {key}: {value:.4f}")
        
        # print other numerical metrics
        for key, value in eval_results.items():
            if (key not in common_metrics and 
                not key.startswith('pass@') and 
                isinstance(value, (int, float)) and 
                key not in ['timestamp', 'prediction_file', 'ground_truth_file', 'evaluation_method']):
                if isinstance(value, float):
                    logger.info(f"   - {key}: {value:.4f}")
                else:
                    logger.info(f"   - {key}: {value}")
    
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
        
        # ensure prediction and ground truth have the same length
        min_len = min(len(predictions), len(references))
        if len(predictions) != len(references):
            logger.warning(f"prediction length ({len(predictions)}) and ground truth length ({len(references)}) do not match, using first {min_len} samples")
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        return predictions, references
    
    def _normalize_answer(self, s: str) -> str:
        """normalize answer text (from TriviaQA)"""
        import string
        import re
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace('_', ' ')

        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()
    
    def _exact_match_score(self, prediction: str, ground_truth: str) -> float:
        """calculate exact match score (from TriviaQA)"""
        return float(self._normalize_answer(prediction) == self._normalize_answer(ground_truth))
    
    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """calculate F1 score (from TriviaQA)"""
        from collections import Counter
        
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def _compute_metrics_with_evaluate(self, predictions: List[str], references: List[str], metrics: List[str]) -> Dict[str, Any]:
        """
        calculate specified metrics, prioritize TriviaQA method for exact_match and f1
        
        Args:
            predictions: prediction result list
            references: ground truth list
            metrics: list of metrics to calculate
            
        Returns:
            evaluation metrics dictionary
        """
        results = {}
        
        for metric_name in metrics:
            try:
                logger.info(f"🔄 calculate metric: {metric_name}")
                
                if metric_name in ['exact_match', 'f1']:
                    # use TriviaQA's calculation method
                    if metric_name == 'exact_match':
                        em_scores = [self._exact_match_score(pred, ref) for pred, ref in zip(predictions, references)]
                        results[metric_name] = sum(em_scores) / len(em_scores) if em_scores else 0.0
                    elif metric_name == 'f1':
                        f1_scores = [self._f1_score(pred, ref) for pred, ref in zip(predictions, references)]
                        results[metric_name] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
                    
                    logger.info(f"✅ {metric_name} calculated: {results[metric_name]:.4f}")
                else:
                    # other metrics use Hugging Face evaluate
                    try:
                        metric = evaluate.load(metric_name)
                        metric_result = metric.compute(predictions=predictions, references=references)
                        results[metric_name] = metric_result
                        logger.info(f"✅ {metric_name} calculated")
                    except Exception as e:
                        logger.error(f"❌ {metric_name} loading failed: {e}")
                        results[metric_name] = None
                
            except Exception as e:
                logger.error(f"❌ {metric_name} calculation failed: {e}")
                results[metric_name] = None
        
        return results
      
    
    def _save_evaluation_results(self, 
                                results: Dict[str, Any], 
                                output_dir: str, 
                                prediction_file: str):
        """
        save evaluation results
        
        Args:
            results: evaluation results
            output_dir: output directory
            prediction_file: prediction file path (for generating file name)
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
    
    def list_available_evaluation_scripts(self) -> Dict[str, List[str]]:
        """
        list available evaluation scripts
        
        Returns:
            evaluation scripts grouped by folder
        """
        evaluation_dir = self.llamafactory_dir / "evaluation"
        available_scripts = {}
        
        if evaluation_dir.exists():
            for item in evaluation_dir.iterdir():
                if item.is_dir():
                    folder_name = item.name
                    scripts = []
                    for script in item.glob("*.py"):
                        if script.name != "__init__.py":
                            scripts.append(f"{folder_name}/{script.name}")
                    if scripts:
                        available_scripts[folder_name] = scripts
        
        return available_scripts
    
    def get_script_usage_info(self, evaluation_script: str) -> Optional[str]:
        """
        get evaluation script usage information (through --help)
        
        Args:
            evaluation_script: evaluation script path
            
        Returns:
            script usage information
        """
        script_path = self.llamafactory_dir / "evaluation" / evaluation_script
        
        if not script_path.exists():
            return None
        
        try:
            result = subprocess.run(
                ["python", str(script_path), "--help"],
                capture_output=True,
                text=True,
                cwd=str(self.llamafactory_dir),
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return None
        except:
            return None
    



# example usage function
def example_usage():
    """example usage method"""
    
    # initialize pipeline
    pipeline = QuestionAnsweringEvaluator(
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
    
    # 1. use template to transform data and generate inference command
    cmd, output_file = pipeline.inference(
        model_path="/path/to/trained/model",
        test_json_file="/path/to/test.json",
        model_config=model_config,
        template_file="evaluation/qa_template_example.py"  # use template to transform
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
    
    # 4. execute evaluation - use TriviaQA specialized evaluation script
    eval_results_triviaqa = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        evaluation_script="triviaqa/triviaqa.py",
        script_args={"mute": False}  # additional parameters
    )
    print(f"evaluation results (TriviaQA script): {eval_results_triviaqa}")
    
    # 5. execute evaluation - use BigCodeBench specialized evaluation script
    eval_results_bigcode = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        evaluation_script="bigcodebench/bigcodebench.py",
        script_args={
            "timeout": 10.0,
            "parallel": 4,
            "pass_k": "1,5,10",
            "calibrated": True
        }
    )
    print(f"evaluation results (BigCodeBench script): {eval_results_bigcode}")
    
    # 6. execute evaluation - specify multiple HF metrics
    eval_results_multi = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        metrics=['exact_match', 'f1', 'bleu', 'rouge', 'meteor']
    )
    print(f"evaluation results (multiple metrics): {eval_results_multi}")
    
    # 7. execute evaluation - only use specific metrics
    eval_results_specific = pipeline.evaluation(
        prediction_file="/path/to/predictions.jsonl",
        ground_truth_file="/path/to/ground_truth.json",
        metrics=['bertscore', 'rouge1', 'rouge2', 'rougeL']
    )
    print(f"evaluation results (specific metrics): {eval_results_specific}")

    # 8. list available evaluation scripts
    available_scripts = pipeline.list_available_evaluation_scripts()
    print(f"available evaluation scripts: {available_scripts}")
    
    # 9. get evaluation script usage information
    usage_info = pipeline.get_script_usage_info("triviaqa/triviaqa.py")
    if usage_info:
        print(f"TriviaQA script usage information:\n{usage_info}")



if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # run example
    example_usage()