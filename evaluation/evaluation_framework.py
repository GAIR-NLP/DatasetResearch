#!/usr/bin/env python3
"""
Complete Evaluation Framework for Deep Dataset
contains three core modules: SFT training, inference and evaluation

features:
- all output files and folders are automatically added timestamp (format: YYYYMMDD_HHMMSS)
- avoid file overwrite, for tracking evaluation results at different times

support two evaluation modes:
1. full pipeline: training → inference → evaluation
2. baseline mode: directly use pre-trained model for inference and evaluation (skip training)

output structure (with timestamp):
saves/run_20240315_143022/model_name/dataset_sft_20240315_143022/
inference_results/run_20240315_143022/dataset_model_inference_20240315_143022.jsonl
evaluation/results/run_20240315_143022/dataset_model_results_20240315_143022.json
evaluation/logs/run_20240315_143022/evaluation_framework_20240315_143022.log

usage examples:
# full pipeline
python evaluation_framework.py --config config.yaml --train_dataset train_data --test_dataset test_data --model llama3_8b

# baseline evaluation
python evaluation_framework.py --config config.yaml --test_dataset test_data --model llama3_8b --baseline

# batch evaluation (support mixed baseline and full pipeline)
python evaluation_framework.py --config config.yaml --step batch --batch_config batch.yaml
"""

import os
import json
import yaml
import subprocess
import argparse
import logging
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import time
import re
from tqdm import tqdm
# import shutil
# import signal
from evaluation.evaluators.classification import ClassificationEvaluator
from evaluation.evaluators.question_answering import QuestionAnsweringEvaluator
from evaluation.evaluators.translation import TranslationEvaluator
from evaluation.evaluators.summarization import SummarizationEvaluator
# from evaluation.classification import ClassificationEvaluator
from evaluation.evaluators.generation import GenerationEvaluator
import threading

# configure logging - initial setup, will be reconfigured later
logger = logging.getLogger(__name__)

class EvaluationFramework:
    """complete evaluation framework, all output files and folders are automatically added timestamp"""
    
    def __init__(self, config_file: str, model_name: str, task_id: str, baseline: bool=False, test: bool=False, base_model: str = 'llama3_8b'):
        """initialize evaluation framework, model_name is the name of search model/generation model, not the object of LLaMA etc. sft"""
        self.config = self.load_config(config_file)
        # generate timestamp for file and folder naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name
        self.task_id = task_id
        self.baseline = baseline
        self.test = test
        self.base_model = base_model
        self.few_shot = self.config['models'][base_model]['few_shot']
        self.n_shot = 'few' if self.config['models'][base_model]['n_shot'] == 3 else self.config['models'][base_model]['n_shot']
        
            
        
        self.workspace = Path(self.config['workspace'])
        self.llamafactory_dir = Path(self.config['llamafactory_dir'])
        self.results_dir = self.llamafactory_dir / "results"
        # 搜索timestamp
        self.search_for_timestamp()

        self.log_dir = self.workspace / "evaluation" / "logs" / f"eval"
        # 设置工作空间
        self.setup_workspace()
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_logging()
        
        # 输出时间戳信息
        logger.info(f"🕐 current evaluation run timestamp: {self.timestamp}")
        logger.info(f"📁 all output files and folders will use this timestamp")
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """load config file"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    def search_for_timestamp(self):
        """
        check if there is a folder named task_id_model_name_timestamp in results directory
        if found, parse timestamp and update self.timestamp
        """
        
        if not os.path.exists(self.results_dir):
            return None
        for dir_name in os.listdir(self.results_dir):
            if self.task_id in dir_name and self.model_name in dir_name:
                # parse timestamp (format: taskid_modelname_timestamp)
                # use regex to match full timestamp format: YYYYMMDD_HHMMSS
                pattern = rf"{re.escape(self.task_id)}_{re.escape(self.model_name)}_(\d{{8}}_\d{{6}})"
                match = re.match(pattern, dir_name)
                if match:
                    found_timestamp = match.group(1)
                    self.timestamp = found_timestamp
                    print(f"✅ found directory: {dir_name}, updated timestamp to: {self.timestamp}")
                    return dir_name
        return None


    def setup_workspace(self):
        """setup workspace"""
        
        # training and inference output are saved in LLaMA-Factory, add timestamp
        if self.test:
            self.models_dir = self.llamafactory_dir / "results" / f"{self.task_id}_{self.model_name}_{self.timestamp}" / "test_saves"
        else:
            self.models_dir = self.llamafactory_dir / "results" / f"{self.task_id}_{self.model_name}_{self.timestamp}" / "saves"
        self.inference_dir = self.llamafactory_dir / "results" / f"{self.task_id}_{self.model_name}_{self.timestamp}"
        self.configs_dir = self.llamafactory_dir / "results" / f"{self.task_id}_{self.model_name}_{self.timestamp}"
        
        # final evaluation results are saved in evaluation directory of workspace, add timestamp
        self.eval_results_dir = self.workspace / "evaluation" / "results" / f"{self.task_id}_{self.model_name}_{self.timestamp}"
        if self.baseline:
            if self.few_shot:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_baseline_{self.n_shot}shot_eval.json"
            else:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_baseline_eval.json"
        elif self.test:
            if self.few_shot:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_test_{self.n_shot}shot_eval.json"
            else:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_test_eval.json"
        else:
            if self.few_shot:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_{self.n_shot}shot_eval.json"
            else:
                self.eval_results_path = self.workspace / "evaluation" / "results" / f"{self.model_name}_eval.json"
        # log file is also saved in evaluation directory of workspace, add timestamp
        
        
        for dir_path in [self.models_dir, self.inference_dir, self.configs_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        if self.test:
            if self.few_shot:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results_test_{self.n_shot}shot.jsonl")
            else:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results_test.jsonl")
        elif self.baseline:
            if self.few_shot:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results_baseline_{self.n_shot}shot.jsonl")
            else:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results_baseline.jsonl")
        else:
            if self.few_shot:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results_{self.n_shot}shot.jsonl")
            else:
                self.inference_path = os.path.join(self.inference_dir, f"inference_results.jsonl")
        if self.test:
            self.configs_path = os.path.join(self.configs_dir, f"test_sft_configs.yaml")
        else:
            self.configs_path = os.path.join(self.configs_dir, f"sft_configs.yaml")
    
    def setup_logging(self):
        """configure logging system"""
        log_file = self.log_dir / f"evaluation_framework.log"
        
        # clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # reconfigure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # force reconfigure
        )
        
    
    def run_command_with_live_output(self, cmd: str, timeout: int = None, description: str = "") -> tuple[int, str]:
        """
        run command and display output in real-time
        
        Args:
            cmd: command to run
            timeout: timeout in seconds
            description: command description
            
        Returns:
            (return_code, stdout_text)
        """
        logger.info(f"start executing {description}: {cmd}")
        logger.info(f"timeout: {timeout} seconds" if timeout else "no timeout")
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        output_lines = []
        start_time = time.time()
        
        try:
            while True:
                # check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"{description} timeout ({timeout} seconds), terminating process...")
                    process.kill()
                    process.wait()
                    return -1, f"Process killed due to timeout ({timeout}s)"
                
                # read output
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    line = output.strip()
                    print(line)  # display to console in real-time
                    output_lines.append(output)
                    # logger.info(line)  # log to file
                
                # short sleep to avoid high CPU usage
                time.sleep(0.01)
            
            return_code = process.poll()
            stdout_text = ''.join(output_lines)
            
            if return_code == 0:
                logger.info(f"{description} executed successfully")
            else:
                logger.error(f"{description} executed failed, return code: {return_code}")
            
            return return_code, stdout_text
            
        except Exception as e:
            logger.error(f"{description} executed failed: {e}")
            try:
                process.kill()
                process.wait()
            except:
                pass
            return -1, str(e)
    
    # ==================== SFT Training Module ====================
    
    def create_sft_config(self, dataset_name: str, model_config: Dict[str, Any]) -> str:
        """create SFT training config file"""
        config_name = f"{dataset_name}_{model_config['name']}_sft_{self.timestamp}.yaml"
        config_path = self.configs_path
        
        # basic SFT config template
        finetuning_type = model_config.get('finetuning_type', 'full')
        if finetuning_type == 'full':
            sft_config = {
                # Model configuration
                'model_name_or_path': model_config['base_model'],
                'trust_remote_code': True,
                
                # Method configuration  
                'stage': 'sft',
                'do_train': True,
                'finetuning_type': model_config.get('finetuning_type', 'full'),
                'deepspeed': 'examples/deepspeed/ds_z3_config.json',
                
                # Dataset configuration
                'dataset': dataset_name,
                'template': model_config.get('template', 'default'),
                'cutoff_len': model_config.get('cutoff_len', 2048),
                'max_samples': model_config.get('max_samples',1000),
                'overwrite_cache': True,
                'preprocessing_num_workers': 16,
                'dataloader_num_workers': 0,
                
                # Output configuration - use LLaMA-Factory standard saves directory structure, add timestamp
                'output_dir': str(self.models_dir),
                'logging_steps': model_config.get('logging_steps', 10),
                'save_steps': model_config.get('save_steps', 1000),
                'plot_loss': True,
                'overwrite_output_dir': True,
                'save_only_model': False,
                'report_to': 'none',
                
                # Training configuration
                'per_device_train_batch_size': model_config.get('batch_size', 1),
                'gradient_accumulation_steps': model_config.get('gradient_accumulation_steps', 2),
                'learning_rate': model_config.get('learning_rate', 1.0e-5),
                'num_train_epochs': model_config.get('num_train_epochs', 3.0),
                'lr_scheduler_type': model_config.get('lr_scheduler_type', 'cosine'),
                'warmup_ratio': model_config.get('warmup_ratio', 0.1),
                'bf16': model_config.get('bf16', True),
                'ddp_timeout': 180000000,
                'resume_from_checkpoint': None,
            }
        else:
            sft_config = {
                # Model configuration
                'model_name_or_path': model_config['base_model'],
                'trust_remote_code': True,
                
                # Method configuration  
                'stage': 'sft',
                'do_train': True,
                'finetuning_type': 'lora',
                'lora_rank': model_config.get('lora_rank', 8),
                'lora_target': model_config.get('lora_target', 'all'),
                
                # Dataset configuration
                'dataset': dataset_name,
                'template': model_config.get('template', 'default'),
                'cutoff_len': model_config.get('cutoff_len', 2048),
                'max_samples': model_config.get('max_samples', None),
                'overwrite_cache': True,
                'preprocessing_num_workers': 16,
                'dataloader_num_workers': 4,
                
                # Output configuration - use LLaMA-Factory standard saves directory structure, add timestamp
                'output_dir': str(self.models_dir),
                'logging_steps': model_config.get('logging_steps', 10),
                'save_steps': model_config.get('save_steps', 1000),
                'plot_loss': True,
                'overwrite_output_dir': True,
                'save_only_model': False,
                'report_to': 'none',
                
                # Training configuration
                'per_device_train_batch_size': model_config.get('batch_size', 1),
                'gradient_accumulation_steps': model_config.get('gradient_accumulation_steps', 8),
                'learning_rate': model_config.get('learning_rate', 1.0e-4),
                'num_train_epochs': model_config.get('num_train_epochs', 3.0),
                'lr_scheduler_type': model_config.get('lr_scheduler_type', 'cosine'),
                'warmup_ratio': model_config.get('warmup_ratio', 0.1),
                'bf16': model_config.get('bf16', True),
                'ddp_timeout': 180000000,
                'resume_from_checkpoint': None,
            }
        
        # add evaluation configuration (if there is validation set)
        if model_config.get('eval_dataset'):
            sft_config.update({
                'eval_dataset': model_config['eval_dataset'],
                'val_size': model_config.get('val_size', 0.1),
                'per_device_eval_batch_size': model_config.get('eval_batch_size', 1),
                'eval_strategy': 'steps',
                'eval_steps': model_config.get('eval_steps', 500),
            })
        
        # save config file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sft_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"SFT config file created: {config_path}")
        return str(config_path)
    
    def run_sft_training(self, dataset_name: str, model_config: Dict[str, Any]) -> Optional[str]:
        """run SFT training"""
        try:
            logger.info(f"start SFT training: {dataset_name} + {model_config['name']}")
            
            # create training config file
            config_path = self.create_sft_config(dataset_name, model_config)
            
            # build training command - need to cd to LLaMA-Factory directory to execute
            relative_config_path = os.path.relpath(config_path, self.llamafactory_dir)
            
            # directly execute cd and llamafactory-cli command through bash
            cmd = f"cd {self.llamafactory_dir} && llamafactory-cli train {relative_config_path}"
            
            logger.info(f"training command: {cmd}")
            
            # run training - display log in real-time
            return_code, stdout_text = self.run_command_with_live_output(
                cmd, 
                timeout=model_config.get('training_timeout', 14400),  # 4 hours timeout
                description="SFT training"
            )
            
            # mock result object to keep compatibility
            class MockResult:
                def __init__(self, returncode, stdout):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = ""
            
            result = MockResult(return_code, stdout_text)
            
            if result.returncode == 0:
                model_path = self.models_dir
                logger.info(f"SFT training completed: {model_path}")
                return str(model_path)
            else:
                logger.error(f"SFT training failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            return None
    
    # ==================== Inference Module ====================
    
    def determine_task_type(self, dataset_name: str) -> str:
        """determine task type: multiple choice or response"""
        # determine from config or dataset name
        if dataset_name in self.config.get('mcq_datasets', []):
            return 'mcq'
        elif any(keyword in dataset_name.lower() for keyword in ['mmlu', 'ceval', 'cmmlu', 'choice']):
            return 'mcq'
        else:
            return 'response'
    
    def create_inference_command(self, model_path: str, test_dataset: str, train_dataset: str, model_config: Dict[str, Any], baseline: bool = False) -> tuple[str, str]:
        """create inference command"""
        # output file path, add timestamp
        suffix = "baseline" if baseline else model_config['name']
        output_file = self.inference_path
        print(f'inference_path: {output_file}')
        
        # basic inference parameters
        if baseline or model_config.get('finetuning_type') != 'lora':
            # baseline mode or full finetuning model directly use model path
            inference_args = [
                f"--model_name_or_path {model_path}",
                f"--dataset {test_dataset}",
                f"--train_dataset {train_dataset}",
                # f"--template {model_config.get('template', 'default')}",
                # f"--cutoff_len {model_config.get('cutoff_len', 2048)}",
                # f"--batch_size {model_config.get('inference_batch_size', 64)}",
                # f"--max_samples {model_config.get('max_samples', 1000)}",
                f"--save_name {output_file}"
            ]
        else:
            # LoRA model need to specify base model and adapter separately
            inference_args = [
                f"--model_name_or_path {model_config['base_model']}",
                f"--adapter_name_or_path {model_path}",
                f"--dataset {test_dataset}",
                f"--train_dataset {train_dataset}",
                # f"--template {model_config.get('template', 'default')}",
                # f"--cutoff_len {model_config.get('cutoff_len', 2048)}",
                # f"--batch_size {model_config.get('inference_batch_size', 64)}",
                # f"--max_samples {model_config.get('max_samples', 1000)}",
                f"--save_name {output_file}"
            ]
        
        
        # build full command
        inference_cmd = f"python scripts/vllm_infer.py {' '.join(inference_args)}"
        full_cmd = f"cd {self.llamafactory_dir} && {inference_cmd}"
        
        logger.info(f"inference command: {full_cmd}")
        return full_cmd, str(output_file)
    
    def run_inference(self, model_path: str, test_dataset: str,train_dataset: str, model_config: Dict[str, Any], evaluator, baseline: bool = False) -> Optional[str]:
        """run inference"""
        suffix = "baseline" if baseline else model_config['name']
        output_file = self.inference_path
        logger.info(f"inference file: {output_file}")
        try:
            logger.info(f"start inference: {model_path} on {test_dataset}")
            
            # determine task type
            if evaluator:
                # create inference command
                cmd, output_file = evaluator.inference(model_path, test_dataset, train_dataset, model_config,output_file)
            else:
                cmd, output_file = self.create_inference_command(model_path, test_dataset, train_dataset, model_config, baseline)
            # run inference command - display log in real-time
            return_code, stdout_text = self.run_command_with_live_output(
                cmd,
                timeout=model_config.get('inference_timeout', 3600),  # 1 hour timeout
                description="模型推理"
            )
            
            # mock result object to keep compatibility
            class MockResult:
                def __init__(self, returncode, stdout):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = ""
            
            result = MockResult(return_code, stdout_text)
            
            if result.returncode == 0:
                logger.info(f"inference completed: {output_file}")
                return output_file
            else:
                logger.error(f"inference failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"inference failed: {e}")
            return None
    
    # ==================== Evaluation Module ====================
    
    def get_evaluation_script(self, dataset_name: str, task_type: str) -> Optional[str]:
        """get evaluation script path"""
        # use dataset specific evaluation script in workspace first
        local_eval_script = self.workspace / "evaluation" / f"{dataset_name}_eval.py"
        if local_eval_script.exists():
            return str(local_eval_script)
        
        # use LLaMA-Factory built-in evaluation script
        llamafactory_eval_script = self.llamafactory_dir / "evaluation" / dataset_name / f"{dataset_name}.py"
        if llamafactory_eval_script.exists():
            return str(llamafactory_eval_script)
        
        # use generic evaluation script in workspace according to task type
        if task_type == 'mcq':
            mcq_eval_script = self.workspace / "evaluation" / "mcq_eval.py"
            if mcq_eval_script.exists():
                return str(mcq_eval_script)
        else:
            response_eval_script = self.workspace / "evaluation" / "response_eval.py"
            if response_eval_script.exists():
                return str(response_eval_script)
        
        logger.warning(f"no evaluation script found for {dataset_name}")
        return None
    
    def run_evaluation(self, inference_file: str, dataset_name: str, task_type: str) -> Optional[Dict[str, Any]]:
        """run evaluation"""
        try:
            logger.info(f"start evaluation: {inference_file}")
            
            # get evaluation script
            eval_script = self.get_evaluation_script(dataset_name, task_type)
            if not eval_script:
                return None
            
            # build evaluation command
            eval_args = [
                "python", eval_script,
                "--prediction_file", inference_file,
                "--output_dir", str(self.eval_results_dir)
            ]
            
            # add dataset specific evaluation parameters
            eval_config = self.config.get('evaluation_configs', {}).get(dataset_name, {})
            for key, value in eval_config.items():
                eval_args.extend([f"--{key}", str(value)])
            
            logger.info(f"evaluation command: {' '.join(eval_args)}")
            
            # run evaluation
            result = subprocess.run(
                eval_args,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                # try to parse JSON output
                try:
                    # usually evaluation script will output JSON result in the last line
                    lines = result.stdout.strip().split('\n')
                    for line in reversed(lines):
                        try:
                            eval_results = json.loads(line)
                            logger.info(f"evaluation completed: {eval_results}")
                            return eval_results
                        except json.JSONDecodeError:
                            continue
                    
                    # if no JSON found, return text output
                    eval_results = {"output": result.stdout, "task_type": task_type}
                    logger.info(f"evaluation completed (text output): {result.stdout}")
                    return eval_results
                except Exception as parse_error:
                    logger.warning(f"parse evaluation result failed: {parse_error}")
                    return {"output": result.stdout, "task_type": task_type}
            else:
                logger.error(f"evaluation failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"evaluation failed: {e}")
            return None
    
    # ==================== Main Pipeline ====================
    
    def save_results(self, dataset_name: str, model_name: str, task_type: str, results: Dict[str, Any]):
        """save evaluation results (append mode)"""
        try:
            result_file = self.eval_results_path
            
            # add meta information
            results_with_meta = {
                "task_id": self.task_id,
                "task_model": self.model_name,
                "dataset_id": dataset_name,
                "baseline": self.baseline,
                "task_type": task_type,
                "results": results
            }
            exist = False
            # read existing content (if any)
            if result_file.exists():
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except Exception:
                    existing = []
            else:
                existing = []
            # if task_id in existing is the same as current task_id, update
            
            for result in existing:
                if result['task_id'] == self.task_id:
                    exist = True
                    result['results'] = results
                    logger.info(f"✅ found existing evaluation results for {self.task_id}, update")
                    break
            # append new result
            if not exist:
                existing.append(results_with_meta)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ results saved: {result_file}")
            
        except Exception as e:
            logger.error(f"save results failed: {e}")
    
    def run_eval_pipeline(self, train_dataset: str, test_dataset: str, model_name: str, task_type: str, metrics=None) -> bool:
        """run evaluation pipeline"""
        self.eval_results_path = Path(str(self.eval_results_path).replace('/results/','/results/final_results/'))
        logger.info(f"task id!!!{self.task_id}")
        if os.path.exists(self.eval_results_path):
            cur_results = json.load(open(self.eval_results_path, 'r'))
            find = False
            for result in cur_results:
                if result['task_id'] == self.task_id:
                    find = True
                    if not ('status' in result['results'].keys()) or (not (result['results']['status'] == 'No Result')):
                        logger.info(f"✅ {self.task_id} evaluation results already exist, skip")
                        return True
                    break
            logger.info(f"find: {find}, task_id {self.task_id}")
        if (train_dataset == 'Search Failed' or train_dataset == 'Generation Failed') and (not self.baseline and not self.test):
            eval_results = {
                'status': 'No Result',
                'reason': f"search or generation failed, {test_dataset}"
            }
            self.save_results(test_dataset, model_name, task_type, eval_results)
            return False
        train_dataset_key = train_dataset
        test_dataset = test_dataset.replace('/','_')
        try:
            if task_type == 'question-answering':
                evaluator = QuestionAnsweringEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'multiple-choice':
                evaluator = ClassificationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'text-classification':
                evaluator = ClassificationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'translation':
                evaluator = TranslationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'summarization':
                evaluator = SummarizationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'text-generation':
                evaluator = GenerationEvaluator(self.workspace, self.llamafactory_dir)
            else:
                logger.error(f"unsupported task type: {task_type}")
                return False
            
            if not evaluator:
                logger.error("evaluation initializer failed, terminate pipeline")
                return False
            logger.info(f"start evaluation pipeline")
            model_config = self.config['models'][model_name]
            inference_file = self.inference_path
            eval_results = None
            if not os.path.exists(inference_file):
                logger.error(f"inference file not found: {inference_file}")
                # determine the reason
                # first exclude the case that there is no file
                
                # 1. check if the key is in dataset_info
                with open(self.llamafactory_dir / 'data' / 'dataset_info.json', 'r') as f:
                    dataset_info = json.load(f)
                if train_dataset_key in dataset_info.keys():
                    inference_file = self.llamafactory_dir / 'data' / dataset_info[train_dataset_key]['file_name']
                    # 2. check if the file exists
                    if os.path.exists(inference_file):
                        # 3. check if the sft failed
                        have_model = False
                        if os.path.exists(self.models_dir):
                            model_exist = False
                            for file in os.listdir(self.models_dir):
                                if file.endswith('.safetensors'):
                                    model_exist = True
                                if ('training_loss.png' in file) and model_exist:
                                    model_path = self.models_dir
                                    logger.info(f"✅ found trained model file, use {model_path}")
                                    have_model = True
                                    break
                                
                        if not have_model:
                            eval_results = {
                                'status': 'No Result',
                                'reason': f'model file not found, {self.models_dir}'
                            }
                        else:
                            eval_results = {
                                        'status': 'No Result',
                                        'reason': 'No specified reason'
                                    }
                    else:
                        eval_results = {
                            'status': 'No Result',
                            'reason': f'dataset file not found, {inference_file}'
                        }
                else:
                    eval_results = {
                        'status': 'No Result',
                        'reason': 'key not in dataset_info',
                        'datasets': f'train: {train_dataset}, test: {test_dataset}, train key: {train_dataset_key}'
                    }
                self.save_results(test_dataset, model_name, task_type, eval_results)
                return False
            else:
                eval_results = evaluator.evaluation(inference_file, test_dataset)
                self.save_results(test_dataset, model_name, task_type, eval_results)
                return True
            
        except Exception as e:
            logger.error(f"evaluation pipeline failed: {e}")
            eval_results = {
                            'status': 'No Result',
                            'reason': 'eval failed'
                        }
            self.save_results(test_dataset, model_name, task_type, eval_results)
            return False
    
    def run_full_pipeline(self, train_dataset: str, test_dataset: str, model_name: str, task_type: str, metrics=None) -> bool:
        """run full evaluation pipeline"""
        try:
            if self.baseline:
                logger.info(f"start baseline evaluation pipeline: {test_dataset} + {model_name} (skip training)")
                # self.inference_path = self.inference_path.replace('.jsonl', '_baseline.jsonl')
            else:
                logger.info(f"start full evaluation pipeline: {train_dataset} → {test_dataset} + {model_name}")
            model_config = self.config['models'][model_name]
            
            print(f'task type: {task_type}')
            if task_type == 'question-answering':
                evaluator = QuestionAnsweringEvaluator(self.workspace, self.llamafactory_dir)
                
            elif task_type == 'multiple-choice':
                evaluator = ClassificationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'text-classification':
                evaluator = ClassificationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'translation':
                evaluator = TranslationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'summarization':
                evaluator = SummarizationEvaluator(self.workspace, self.llamafactory_dir)
            elif task_type == 'text-generation':
                evaluator = GenerationEvaluator(self.workspace, self.llamafactory_dir)
            else:
                logger.error(f"unsupported task type: {task_type}")

                evaluator = None
            
            if not evaluator:
                if self.baseline:
                    # Baseline mode: use original pre-trained model, skip training
                    logger.info("=== Baseline mode: use original pre-trained model ===")
                    model_path = model_config['base_model']
                else:
                    # 1. SFT training
                    logger.info("=== step 1: SFT training ===")
                    # first check if the model is already trained
                    
                    use_model = False
                    # check if there is a .safetensors file in models_dir
                    if os.path.exists(self.models_dir):
                        model_exist = False
                        for file in os.listdir(self.models_dir):
                            if file.endswith('.safetensors'):
                                model_exist = True
                                logger.info(f"✅ found a model file: {file}")
                                continue
                            if ('training_loss.png' in file) and model_exist:
                                model_path = self.models_dir
                                logger.info(f"✅ found trained model file, use {model_path}")
                                use_model = True
                                break
                    if not use_model:
                        logger.info("🔍 no trained model found, start training...")
                        model_path = self.run_sft_training(train_dataset, model_config)
                    if not model_path:
                        logger.error("SFT training failed, terminate pipeline")
                        return False
                
                # 2. inference
                step_num = "step 1" if self.baseline else "step 2"
                logger.info(f"=== {step_num}: inference ===")
                # first check if there is already a inference result
                inference_file = None
                if os.path.exists(self.inference_path):
                    logger.info(f"✅ found existing inference result, use {self.inference_path}")
                    inference_file = self.inference_path
                else:
                    logger.info(f"🔍 no existing inference result found, start inference...")
                    inference_file = self.run_inference(model_path, test_dataset, train_dataset, model_config, evaluator, self.baseline)
                if not inference_file:
                    logger.error("inference failed, terminate pipeline")
                    return False
                
                # 3. evaluation
                step_num = "step 2" if self.baseline else "step 3"
                logger.info(f"=== {step_num}: evaluation ===")
                eval_results = self.run_evaluation(inference_file, test_dataset, task_type)

                if eval_results is None:
                    logger.error("evaluation failed, terminate pipeline")
                    return False
                
                # 4. save results
                step_num = "step 3" if self.baseline else "step 4"
                logger.info(f"=== {step_num}: save results ===")
                result_key = f"{test_dataset}"
                self.save_results(result_key, model_name, task_type, eval_results)
                
                logger.info("✅ full evaluation pipeline completed!")
                return True
            else:
                if self.baseline:
                    # Baseline mode: use original pre-trained model, skip training
                    logger.info("=== Baseline mode: use original pre-trained model ===")
                    model_path = model_config['base_model']
                else:
                    # 1. SFT training
                    logger.info("=== step 1: SFT training ===")
                    # first check if there is already a trained model
                    
                    use_model = False
                    # check if there is a .safetensors file in models_dir
                    if os.path.exists(self.models_dir):
                        model_exist = False
                        for file in os.listdir(self.models_dir):
                            if file.endswith('.safetensors'):
                                model_exist = True
                                logger.info(f"✅ found a model file: {file}")
                                continue
                            if ('training_loss.png' in file) and model_exist:
                                model_path = self.models_dir
                                logger.info(f"✅ found trained model file, use {model_path}")
                                use_model = True
                                break
                    if not use_model:
                        logger.info("🔍 no trained model found, start training...")
                        model_path = self.run_sft_training(train_dataset, model_config)
                    if not model_path:
                        logger.error("SFT training failed, terminate pipeline")
                        return False
                    # model_path = "./saves/llama3_8b/iamtarun_python_code_instructions_18k_alpaca_sft"
                
                # 2. inference
                step_num = "step 1" if self.baseline else "step 2"
                logger.info(f"=== {step_num}: inference ===")
                # first check if there is already a inference result
                inference_file = None
                if os.path.exists(self.inference_path):
                    logger.info(f"✅ found existing inference result, use {self.inference_path}")
                    inference_file = self.inference_path
                else:
                    logger.info(f"🔍 no existing inference result found, start inference...")
                    inference_file = self.run_inference(model_path, test_dataset,train_dataset, model_config, evaluator, self.baseline)
                
                if not inference_file:
                    logger.error("inference failed, terminate pipeline")
                    return False
                
                # 3. evaluation
                step_num = "step 2" if self.baseline else "step 3"
                logger.info(f"=== {step_num}: evaluation ===")
                
                # first check if there is already a evaluation result
                eval_result_file = self.eval_results_path
                # eval_results = None
                eval_exist = False
                if eval_result_file.exists():
                    try:
                        with open(eval_result_file, 'r', encoding='utf-8') as f:
                            existing_results = json.load(f)
                        # check if the evaluation result contains the current dataset and model
                        if isinstance(existing_results, list):
                            for result in existing_results:
                                if (result.get('task_id') == self.task_id and 
                                    result.get('task_model') == self.model_name):
                                    eval_exist = True
                                    logger.info(f"✅ found existing evaluation result, skip")
                                    break
                    except Exception as e:
                        logger.warning(f"⚠️ read existing evaluation result failed: {e}")
                
                if eval_exist is False:
                    logger.info("🔍 no existing evaluation result found, start evaluation...")
                    # eval_results = self.run_evaluation(inference_file, test_dataset, task_type)
                    # add metrics through config
                    # metrics = model_config.get('metrics', None)
                    eval_results = None
                    # if metrics is None:
                    #     eval_results = evaluator.evaluation(inference_file, test_dataset)
                    # else:
                    #     eval_results = evaluator.evaluation(inference_file, test_dataset, metrics=metrics)
                    if eval_results is None:
                        logger.error("evaluation failed, terminate pipeline")
                        return False
                
                # 4. save results
                    step_num = "step 3" if self.baseline else "step 4"
                    logger.info(f"=== {step_num}: save results ===")
                    if self.baseline:
                        result_key = f"{test_dataset}"
                    else:
                        result_key = f"{test_dataset}"
                    self.save_results(result_key, model_name, task_type, eval_results)
                
                logger.info("✅ full evaluation pipeline completed!")
                return True
            
        except Exception as e:
            logger.error(f"evaluation pipeline failed: {e}")
            return False
    
    def run_batch_evaluation(self, batch_config: List[Dict[str, str]]) -> Dict[str, bool]:
        """run batch evaluation"""
        results = {}
        for config in batch_config:
            train_dataset = config.get('train_dataset', '')  # baseline mode may be empty
            test_dataset = config['test_dataset']
            model_name = config['model']
            baseline = config.get('baseline', False)
            
            if baseline:
                key = f"baseline_{test_dataset}+{model_name}"
                logger.info(f"start batch baseline evaluation: {key}")
                success = self.run_full_pipeline(train_dataset, test_dataset, model_name, baseline=True)
            else:
                key = f"{train_dataset}→{test_dataset}+{model_name}"
                logger.info(f"start batch evaluation: {key}")
                success = self.run_full_pipeline(train_dataset, test_dataset, model_name, baseline=False)
            
            results[key] = success
            
            logger.info(f"batch evaluation result: {key} = {'success' if success else 'failed'}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Deep Dataset Evaluation Framework")
    parser.add_argument("--config", required=True, help="config file path")
    # parser.add_argument("--train_dataset", help="train dataset name")
    # parser.add_argument("--test_dataset", help="test dataset name")
    parser.add_argument("--dataset_json", help="dataset metadata json file path")
    parser.add_argument("--model", default='llama3_8b', help="model name")
    parser.add_argument("--step", choices=['train', 'inference', 'eval', 'full', 'batch'], 
                       default='full', help="run step")
    parser.add_argument("--task_model",default='gpt-4o-search',help="model name")
    # parser.add_argument("--batch_config", help="batch evaluation config file")
    parser.add_argument("--baseline", action='store_true', help="run baseline evaluation (skip training, use pre-trained model)")
    parser.add_argument("--test", action='store_true', help="run test set evaluation (skip training, use pre-trained model)")
    
    args = parser.parse_args()
    # create evaluation framework
    print('start evaluation')
    metadatas = json.load(open(args.dataset_json, 'r', encoding='utf-8'))
    for metadata in tqdm(metadatas):
        # train_dataset = metadata['dataset_id']
        test_dataset = metadata['original_dataset_id'].replace('/','_')
        if 'task_id' in metadata.keys():
            logger.info(f"Have task_id")
            task_id = metadata['task_id']
        else:
            task_id = test_dataset
        print(f"task_id: {task_id}")
        if args.test:
            train_dataset = test_dataset
        else:
            # if 'Failed' in metadata['search_dataset_id'] and (not args.step == 'eval' and not args.baseline and not args.test):
            #     continue
            train_dataset = metadata['search_dataset_id']
                

        print(f"start evaluation: train_dataset: {train_dataset}, test_dataset: {test_dataset}")
        task_type = metadata['original_metadata']['task']
        
        try:
            framework = EvaluationFramework(args.config, args.task_model, task_id, args.baseline, args.test,args.model)
            if args.step == 'full':
                
                if args.baseline:
                    # Baseline mode only needs test_dataset and model
                    if not all([test_dataset, args.model]):
                        print("Baseline evaluation needs --model")
                        return
                    print(f"🎯 baseline evaluation: {test_dataset} + {args.model}")
                else:
                    # full pipeline needs all parameters
                    if not all([train_dataset, test_dataset, args.model]):
                        print("full pipeline needs --train_dataset, --test_dataset, --model")
                        return
                    print(f"🎯 start full pipeline: {train_dataset} → {test_dataset} + {args.model}")
                
                success = framework.run_full_pipeline(
                    train_dataset or "", 
                    test_dataset, 
                    args.model, 
                    task_type
                )
                
                if args.baseline:
                    print("✅ baseline evaluation completed" if success else "❌ baseline evaluation failed")
                else:
                    print("✅ full evaluation pipeline completed" if success else "❌ full evaluation pipeline failed")
            elif args.step == 'eval':
                success = framework.run_eval_pipeline(
                    metadata['search_dataset_id'], 
                    metadata['original_dataset_id'], 
                    args.model, 
                    task_type
                )
                print("✅ evaluation pipeline completed" if success else "❌ evaluation pipeline failed")
            else:
                logger.error(f'error step for train_dataset: {train_dataset}, test_dataset: {test_dataset}"')
        except Exception as e:
            logger.error(f'error: {e} for train_dataset: {train_dataset}, test_dataset: {test_dataset}"')
            print(f"error: {e}")
            continue

if __name__ == "__main__":
    main() 