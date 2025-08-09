#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated evaluation pipeline script
features:
1. orchestrate evaluation_framework.py and process_eval_results.py
2. maintain separation of concerns: evaluation execution vs result processing
3. follow the established architecture patterns
"""

import os
import sys
sys.path.append(os.getcwd())
import yaml
import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import numpy as np
from collections import defaultdict

# import functions from process_eval_results.py
sys.path.append(os.path.join(os.getcwd(), 'DatasetResearch', 'evaluation'))
from evaluation.process_eval_results import (
    load_evaluation_data,
    process_evaluation_data, 
    TASK_TYPE_METRICS,
    DEFAULT_METRICS,
    extract_metric_score
)


class IntegratedEvaluationPipeline:
    """Orchestrator for evaluation workflow following the established architecture"""
    
    def __init__(self, eval_config_path: str, pipeline_config_path: str, base_dir: str = None, dataset_set: str = "full"):
        """
        initialize the integrated evaluation pipeline
        
        Args:
            eval_config_path: evaluation config file path (evaluation/config.yaml)
            pipeline_config_path: pipeline config file path (configs/pipeline_settings.yaml)
            base_dir: project base directory path
            dataset_set: which dataset set to use ("full" or "mini")
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.eval_config_path = Path(eval_config_path)
        self.pipeline_config_path = Path(pipeline_config_path)
        self.dataset_set = dataset_set
        
        # load two config files
        self.eval_config = self._load_yaml_config(self.eval_config_path)
        self.pipeline_config = self._load_yaml_config(self.pipeline_config_path)
        
        # get path information from evaluation config
        self.workspace = Path(self.eval_config.get('workspace', self.base_dir))
        self.llamafactory_dir = Path(self.eval_config.get('llamafactory_dir', self.base_dir / 'LLaMA-Factory'))
        
        # set directory paths
        self.eval_dir = self.workspace / "evaluation"
        self.results_dir = self.eval_dir / "results" / "final_results"
        
        # ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # set up logging
        self.setup_logging()
        
    def _load_yaml_config(self, config_path: Path) -> Dict:
        """load YAML config file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ config file loaded: {config_path}")
            return config
        except Exception as e:
            print(f"❌ failed to load config file {config_path}: {e}")
            sys.exit(1)
        
    def setup_logging(self):
        """set up logging system"""
        log_file = self.eval_dir / "logs" / "integrated_pipeline.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # clear existing handlers
        logger = logging.getLogger(__name__)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def get_model_configs(self) -> Dict[str, Dict]:
        """get model configs from evaluation config"""
        return self.eval_config.get('models', {})
    
    def get_target_data_models(self) -> Dict[str, Dict]:
        """get target data models from pipeline config"""
        return self.pipeline_config.get('target_data_models', {})
    
    def get_evaluation_methods(self) -> Dict[str, Dict]:
        """get evaluation methods from pipeline config"""
        return self.pipeline_config.get('evaluation_settings', {}).get('methods', {})

    def find_dataset_files(self) -> List[Path]:
        """find dataset JSON files according to pipeline config"""
        dataset_config = self.pipeline_config.get('dataset_files', {})
        
        if dataset_config.get('auto_discover', True):
            # auto-discovery mode
            search_path = dataset_config.get('search_path', 'datasets/results/')
            datasets_dir = self.base_dir / search_path
            
            if not datasets_dir.exists():
                return []
            
            patterns = dataset_config.get('patterns', [
                "search_*.json",
                "generation_*.json"
            ])
            
            found_files = []
            for pattern in patterns:
                found_files.extend(datasets_dir.glob(pattern))
            
            return sorted(found_files)
        else:
            # manual mode
            search_path = dataset_config.get('search_path', 'datasets/results/')
            datasets_dir = self.base_dir / search_path
            manual_files = dataset_config.get('manual_files', [])
            return [datasets_dir / f for f in manual_files if (datasets_dir / f).exists()]

    def check_reference_files(self) -> bool:
        """check if reference files exist"""
        test_file = self.results_dir / "final_test.json"
        baseline_file = self.results_dir / "final_baseline.json"
        
        if test_file.exists() and baseline_file.exists():
            self.logger.info("reference files exist")
            return True
        
        self.logger.warning("missing reference files")
        if not test_file.exists():
            self.logger.warning(f"missing: {test_file}")
        if not baseline_file.exists():
            self.logger.warning(f"missing: {baseline_file}")
        
        return False

    def run_evaluation_framework(self, dataset_json: Path, task_model: str, base_model: str, 
                                baseline: bool = False, test: bool = False) -> bool:
        """
        run evaluation using EvaluationFramework
        
        Args:
            dataset_json: dataset JSON file path
            task_model: task/search model name
            base_model: base SFT model name
            baseline: whether to run baseline evaluation
            test: whether to run test evaluation
            
        Returns:
            whether successful
        """
        try:
            self.logger.info(f"running evaluation: {dataset_json.name}, baseline={baseline}, test={test}")
            
            # use EvaluationFramework directly following its established pattern
            cmd = [
                "python", "evaluation/evaluation_framework.py",
                "--config", str(self.eval_config_path),
                "--dataset_json", str(dataset_json),
                "--model", base_model,
                "--task_model", task_model,
                "--step", "eval"
            ]
            
            if baseline:
                cmd.append("--baseline")
            if test:
                cmd.append("--test")
            
            self.logger.info(f"executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("evaluation framework completed successfully")
                return True
            else:
                self.logger.error(f"evaluation framework failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"evaluation framework failed: {e}")
            return False

    def _is_valid_filename(self, filename: str) -> bool:
        """
        validate filename format according to config-defined patterns and task count matching
        
        Args:
            filename: evaluation result filename
            
        Returns:
            whether filename is valid
        """
        # get target models and methods from config
        target_models = self.get_target_data_models()
        evaluation_settings = self.pipeline_config.get('evaluation_settings', {})
        methods_config = evaluation_settings.get('methods', {})
        
        # remove .json extension if present
        name = filename.replace('.json', '')
        
        # first check filename pattern
        pattern_valid = False
        for method_key, method_config in methods_config.items():
            filename_patterns = method_config.get('filename_patterns', [])
            
            for pattern in filename_patterns:
                # handle patterns without {model_key} placeholder
                if '{model_key}' not in pattern:
                    if name == pattern or name == f"final_{pattern}":
                        pattern_valid = True
                        break
                else:
                    # handle patterns with {model_key} placeholder
                    for model_key in target_models.keys():
                        # try both with and without _eval suffix
                        expected_filename1 = pattern.replace('{model_key}', model_key)
                        expected_filename2 = f"{pattern.replace('{model_key}', model_key)}_eval"
                        if name == expected_filename1 or name == expected_filename2:
                            pattern_valid = True
                            break
            if pattern_valid:
                break
        
        if not pattern_valid:
            return False
        
        # for evaluation files (not final_baseline/final_test), validate task count
        if name not in ['final_baseline', 'final_test']:
            file_path = self.results_dir / filename
            if file_path.exists():
                try:
                    # load evaluation data to get actual task count
                    eval_data = self._load_evaluation_data(str(file_path))
                    actual_task_count = len(eval_data)
                    
                    # get expected task count by loading the selected metadata file
                    expected_task_count = self._get_expected_task_count_from_metadata()
                    
                    if actual_task_count < expected_task_count:
                        self.logger.warning(f"{filename} has {actual_task_count} tasks, less than expected {expected_task_count} for {self.dataset_set} set")
                        return False
                    
                    self.logger.info(f"{filename} has {actual_task_count} tasks, valid for {self.dataset_set} set (expected: {expected_task_count})")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to validate task count for {filename}: {e}")
                    return False
        
        return True
    
    def _get_expected_task_count_from_metadata(self) -> int:
        """
        get expected task count by loading the selected metadata file
        
        Returns:
            expected task count
        """
        try:
            # get domain metadata file path based on dataset_set
            dataset_config = self.pipeline_config.get('dataset_files', {})
            domain_metadata_files = dataset_config.get('domain_metadata_files', {})
            
            config_key = f"{self.dataset_set}_set"
            if config_key not in domain_metadata_files:
                self.logger.warning(f"Dataset set '{config_key}' not found in config")
                return 0
                
            domain_metadata_file = domain_metadata_files[config_key].get('file_path')
            if not domain_metadata_file:
                self.logger.warning(f"No file_path found for {config_key}")
                return 0
                
            # load metadata file and count tasks
            full_path = self.base_dir / domain_metadata_file
            if not full_path.exists():
                self.logger.warning(f"Metadata file not found: {full_path}")
                return 0
                
            with open(full_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            task_count = len(metadata)
            self.logger.info(f"Loaded metadata from {domain_metadata_file}: {task_count} tasks")
            return task_count
            
        except Exception as e:
            self.logger.warning(f"Failed to get expected task count from metadata: {e}")
            return 0

    def process_evaluation_results(self, test_file: Path, baseline_file: Path, 
                                 eval_files: List[Path], output_dir: Path) -> bool:
        """
        process evaluation results and generate CSV output in required format
        
        Args:
            test_file: test reference file
            baseline_file: baseline reference file
            eval_files: evaluation result files
            output_dir: output directory
            
        Returns:
            whether successful
        """
        try:
            self.logger.info(f"processing evaluation results for {len(eval_files)} files")
            
            # get output configuration from pipeline_settings.yaml
            output_config = self.pipeline_config.get('output', {})
            csv_filename = output_config.get('csv_filename', 'evaluation_summary.csv')
            save_path = output_config.get('save_path', 'evaluation/results/')
            format_config = output_config.get('format', {})
            decimal_places = format_config.get('decimal_places', 4)
            include_std = format_config.get('include_std', True)
            baseline_row_config = output_config.get('baseline_row', {})
            
            # use configured output directory
            if not output_dir.is_absolute():
                output_dir = self.base_dir / save_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # collect all results for CSV output
            all_results = []
            
            # load test and baseline data once
            test_data = self._load_evaluation_data(str(test_file))
            baseline_data = self._load_evaluation_data(str(baseline_file))
            
            if not test_data or not baseline_data:
                self.logger.error("cannot load test.json or baseline.json file")
                return False
            
            # process each evaluation file - filter valid filenames only
            valid_files = []
            for eval_file in eval_files:
                if eval_file.name in ["final_test.json", "final_baseline.json"]:
                    continue
                
                # validate filename format
                if not self._is_valid_filename(eval_file.name):
                    self.logger.warning(f"skipping invalid filename: {eval_file.name}")
                    continue
                
                valid_files.append(eval_file)
            
            self.logger.info(f"processing {len(valid_files)} valid files (filtered from {len(eval_files)} total files)")
            
            for eval_file in valid_files:
                self.logger.info(f"processing: {eval_file.name}")
                
                # extract model and method from filename using config mappings
                model, method = self._parse_filename(eval_file.stem)
                
                # debug info
                self.logger.info(f"parsed {eval_file.stem}: model={model}, method={method}")
                
                # load evaluation data
                eval_data = self._load_evaluation_data(str(eval_file))
                if not eval_data:
                    self.logger.warning(f"failed to load {eval_file.name}")
                    continue
                
                # calculate domain scores
                domain_scores = self._calculate_domain_scores(test_data, baseline_data, eval_data)
                
                # create result row with configured format
                result_row = {
                    'Model': model,
                    'Method': method,
                    'knowledge': round(domain_scores.get('knowledge', {}).get('mean', 0.0), decimal_places),
                    'reasoning': round(domain_scores.get('reasoning', {}).get('mean', 0.0), decimal_places),
                    'agent': round(domain_scores.get('agent', {}).get('mean', 0.0), decimal_places)
                }
                
                # add standard deviation columns if configured
                if include_std:
                    result_row.update({
                        'knowledge_std': round(domain_scores.get('knowledge', {}).get('std', 0.0), decimal_places),
                        'reasoning_std': round(domain_scores.get('reasoning', {}).get('std', 0.0), decimal_places),
                        'agent_std': round(domain_scores.get('agent', {}).get('std', 0.0), decimal_places)
                    })
                
                all_results.append(result_row)
                self.logger.info(f"processed {eval_file.name}: {model}, {method}")
            
            # add baseline row if configured - calculate actual baseline scores
            if baseline_row_config.get('position') == 'first':
                baseline_model_name = baseline_row_config.get('model_name', 'LLaMA-3.1-8B')
                baseline_method_name = baseline_row_config.get('method_name', 'baseline')
                
                # calculate baseline domain scores using actual baseline data
                baseline_domain_scores = self._calculate_domain_scores(test_data, baseline_data, baseline_data)
                
                # create baseline row with actual calculated scores
                baseline_row = {
                    'Model': baseline_model_name,
                    'Method': baseline_method_name,
                    'knowledge': round(baseline_domain_scores.get('knowledge', {}).get('mean', 0.0), decimal_places),
                    'reasoning': round(baseline_domain_scores.get('reasoning', {}).get('mean', 0.0), decimal_places),
                    'agent': round(baseline_domain_scores.get('agent', {}).get('mean', 0.0), decimal_places)
                }
                
                if include_std:
                    baseline_row.update({
                        'knowledge_std': round(baseline_domain_scores.get('knowledge', {}).get('std', 0.0), decimal_places),
                        'reasoning_std': round(baseline_domain_scores.get('reasoning', {}).get('std', 0.0), decimal_places),
                        'agent_std': round(baseline_domain_scores.get('agent', {}).get('std', 0.0), decimal_places)
                    })
                
                # insert baseline row at the beginning
                all_results.insert(0, baseline_row)
            
            # create final CSV
            if all_results:
                results_df = pd.DataFrame(all_results)
                
                # sort by model name to group same models together
                # baseline stays at top, then sort others by model
                baseline_rows = results_df[results_df['Model'].str.contains('baseline', case=False, na=False)]
                other_rows = results_df[~results_df['Model'].str.contains('baseline', case=False, na=False)]
                other_rows_sorted = other_rows.sort_values('Model')
                
                # combine: baseline first, then sorted by model
                results_df = pd.concat([baseline_rows, other_rows_sorted], ignore_index=True)
                
                csv_file = output_dir / csv_filename
                results_df.to_csv(csv_file, index=False)
                self.logger.info(f"CSV results saved to: {csv_file}")
                
                # also print results to console
                print("\n📊 Evaluation Results:")
                print("=" * 80)
                float_format = f'%.{decimal_places}f'
                print(results_df.to_string(index=False, float_format=float_format))
                
                return True
            else:
                self.logger.warning("no results to save")
                return False
            
        except Exception as e:
            self.logger.error(f"process_evaluation_results failed: {e}")
            return False

    def run_integrated_pipeline(self, run_evaluation: bool = True,
                              process_results: bool = True) -> bool:
        """
        run integrated evaluation pipeline
        
        Args:
            dataset_files: dataset file list
            task_models: task model list  
            base_models: base model list
            run_evaluation: whether to run evaluation
            process_results: whether to process results
            
        Returns:
            whether successful
        """
        self.logger.info("starting integrated evaluation pipeline")
        
        # get configuration from pipeline config
        evaluation_settings = self.pipeline_config.get('evaluation_settings', {})
        base_model = evaluation_settings.get('sft_model', 'llama3_8b')
        
        # get target data models from pipeline config
        target_data_models = self.get_target_data_models()
        task_models = list(target_data_models.keys())
        
        # get dataset files
        dataset_paths = self.find_dataset_files()
        
        self.logger.info(f"target base model: {base_model}")
        self.logger.info(f"target task models: {task_models}")
        self.logger.info(f"dataset files: {[p.name for p in dataset_paths]}")
        
        success = True
        
        # step 1: run evaluations
        if run_evaluation:
            self.logger.info("=== step 1: running evaluations ===")
            
            for dataset_path in dataset_paths:
                for task_model in task_models:
                    # run test evaluation to generate reference
                    if not self.run_evaluation_framework(
                        dataset_path, task_model, base_model, test=True):
                        self.logger.warning(f"test evaluation failed: {dataset_path.name}")
                        success = False
                    
                    # run baseline evaluation to generate reference
                    if not self.run_evaluation_framework(
                        dataset_path, task_model, base_model, baseline=True):
                        self.logger.warning(f"baseline evaluation failed: {dataset_path.name}")
                        success = False
                    
                    # run full evaluation
                    if not self.run_evaluation_framework(
                        dataset_path, task_model, base_model):
                        self.logger.warning(f"full evaluation failed: {dataset_path.name}")
                        success = False
        
        # step 2: process results
        if process_results:
            self.logger.info("=== step 2: processing results ===")
            
            # check reference files
            if not self.check_reference_files():
                self.logger.error("missing reference files, cannot process results")
                return False
            
            # find all result files
            result_files = list(self.results_dir.glob("*.json"))
            if not result_files:
                self.logger.warning("no result files found")
                return success
            
            # process results
            test_file = self.results_dir / "final_test.json"
            baseline_file = self.results_dir / "final_baseline.json"
            output_dir = self.eval_dir / "results" / "processed"
            
            if not self.process_evaluation_results(
                test_file, baseline_file, result_files, output_dir):
                success = False
        
        if success:
            self.logger.info("integrated pipeline completed successfully")
        else:
            self.logger.warning("integrated pipeline completed with warnings")
        
        return success

    def _load_evaluation_data(self, file_path: str) -> Dict[str, Dict]:
        """
        load evaluation data file using process_eval_results.load_evaluation_data
        
        Args:
            file_path: JSON file path
        
        Returns:
            dictionary containing task_id and detailed evaluation data
        """
        return load_evaluation_data(file_path)
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        parse model and method from filename using ONLY config mappings
        
        Args:
            filename: evaluation result filename
            
        Returns:
            (model, method) tuple
        """
        # get all configuration
        target_models = self.get_target_data_models()
        evaluation_settings = self.pipeline_config.get('evaluation_settings', {})
        methods_config = evaluation_settings.get('methods', {})
        sft_model_display_name = evaluation_settings.get('sft_model_display_name', 'LLaMA-3.1-8B')
        
        # clean filename - remove common prefixes/suffixes
        clean_name = filename.replace('final_', '').replace('_eval', '').replace('.json', '')
        
        # try to match against all method patterns
        for method_key, method_config in methods_config.items():
            filename_patterns = method_config.get('filename_patterns', [])
            method_display_name = method_config.get('display_name', method_key)
            
            for pattern in filename_patterns:
                # handle patterns without {model_key} placeholder
                if '{model_key}' not in pattern:
                    if clean_name == pattern.replace('final_', ''):
                        # special files like final_baseline, final_test
                        return sft_model_display_name, method_display_name
                else:
                    # handle patterns with {model_key} placeholder
                    for model_key, model_config in target_models.items():
                        expected_name = pattern.replace('{model_key}', model_key).replace('final_', '').replace('_eval', '')
                        if clean_name == expected_name:
                            model_display_name = model_config.get('display_name', model_key)
                            return model_display_name, method_display_name
        
        # fallback: if no pattern matches, use default
        self.logger.warning(f"No pattern matched for filename: {filename}, using defaults")
        return sft_model_display_name, methods_config.get('baseline', {}).get('display_name', 'baseline')
    
    def _extract_domain_from_task_id(self, task_id: str) -> str:
        """
        extract domain from task_id based on patterns
        
        Args:
            task_id: task identifier
            
        Returns:
            domain (knowledge, reasoning, or agent)
        """
        task_id_lower = task_id.lower()
        
        # knowledge domain patterns
        knowledge_patterns = [
            'knowledge', 'qa', 'question', 'answer', 'reading', 'comprehension',
            'factual', 'trivia', 'encyclopedia', 'wiki', 'fact', 'information',
            'recall', 'memory', 'general-knowledge', 'commonsense-qa'
        ]
        
        # reasoning domain patterns  
        reasoning_patterns = [
            'reasoning', 'logic', 'mathematical', 'math', 'calculation', 'problem',
            'analytical', 'deductive', 'inductive', 'inference', 'puzzle',
            'legal-reasoning', 'mathematical-reasoning', 'logical-reasoning',
            'causal-reasoning', 'abstract-reasoning'
        ]
        
        # agent domain patterns
        agent_patterns = [
            'agent', 'action', 'planning', 'decision', 'strategy', 'simulation',
            'environment', 'game', 'navigation', 'control', 'policy', 'execution',
            'tool', 'function', 'api', 'workflow', 'task-planning'
        ]
        
        # check patterns in order of specificity
        for pattern in reasoning_patterns:
            if pattern in task_id_lower:
                return 'reasoning'
                
        for pattern in knowledge_patterns:
            if pattern in task_id_lower:
                return 'knowledge'
                
        for pattern in agent_patterns:
            if pattern in task_id_lower:
                return 'agent'
        
        # fallback: try to guess from other indicators
        if any(word in task_id_lower for word in ['cot', 'chain-of-thought', 'step-by-step']):
            return 'reasoning'
        elif any(word in task_id_lower for word in ['generation', 'completion', 'summary']):
            return 'knowledge'
        else:
            return 'agent'  # default fallback
    
    def _load_domain_mapping(self, json_file: str) -> Dict[str, str]:
        """
        load task_id to domain mapping from JSON file
        
        Args:
            json_file: JSON file containing original_metadata
        
        Returns:
            dictionary containing task_id to domain mapping
        """
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            domain_mapping = {}
            for item in data:
                task_id = item.get('task_id')
                if task_id:
                    original_metadata = item.get('original_metadata', {})
                    if isinstance(original_metadata, dict):
                        domain = original_metadata.get('domain', 'unknown')
                    else:
                        domain = 'unknown'
                    domain_mapping[task_id] = domain
            
            self.logger.info(f"✅ successfully loaded domain mapping from {json_file}, contains {len(domain_mapping)} task_ids")
            return domain_mapping
        except Exception as e:
            self.logger.warning(f"❌ load domain mapping failed: {e}")
            return {}
    
    def _calculate_domain_scores(self, test_data: Dict, baseline_data: Dict, eval_data: Dict) -> Dict[str, Dict]:
        """
        calculate scores by domain using process_eval_results.py logic
        
        Args:
            test_data: test reference data
            baseline_data: baseline reference data
            eval_data: evaluation data
            
        Returns:
            domain scores with mean and std
        """
        # get domain metadata file based on dataset_set parameter
        dataset_config = self.pipeline_config.get('dataset_files', {})
        domain_metadata_files = dataset_config.get('domain_metadata_files', {})
        
        # select domain metadata file based on dataset_set
        domain_metadata_file = None
        config_key = f"{self.dataset_set}_set"
        if config_key in domain_metadata_files:
            domain_metadata_file = domain_metadata_files[config_key].get('file_path')
            self.logger.info(f"Using {self.dataset_set} domain metadata file: {domain_metadata_file}")
        else:
            self.logger.warning(f"Dataset set '{config_key}' not found in config, using fallback")
            # fallback to first available metadata file
            if domain_metadata_files:
                first_set = next(iter(domain_metadata_files.values()))
                domain_metadata_file = first_set.get('file_path')
                self.logger.info(f"Using fallback domain metadata file: {domain_metadata_file}")
        
        # write temp eval file for process_evaluation_data function
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_eval_file:
            # convert eval_data to list format expected by process_evaluation_data
            eval_list = []
            for task_id, data in eval_data.items():
                eval_list.append({
                    'task_id': task_id,
                    'task_type': data.get('task_type', 'unknown'),
                    'results': data.get('results', {}),
                    'baseline': data.get('baseline', False)
                })
            json.dump(eval_list, temp_eval_file, ensure_ascii=False, indent=2)
            temp_eval_path = temp_eval_file.name
        
        # write temp test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_test_file:
            test_list = []
            for task_id, data in test_data.items():
                test_list.append({
                    'task_id': task_id,
                    'task_type': data.get('task_type', 'unknown'),
                    'results': data.get('results', {}),
                    'baseline': data.get('baseline', False)
                })
            json.dump(test_list, temp_test_file, ensure_ascii=False, indent=2)
            temp_test_path = temp_test_file.name
        
        # write temp baseline file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_baseline_file:
            baseline_list = []
            for task_id, data in baseline_data.items():
                baseline_list.append({
                    'task_id': task_id,
                    'task_type': data.get('task_type', 'unknown'),
                    'results': data.get('results', {}),
                    'baseline': data.get('baseline', False)
                })
            json.dump(baseline_list, temp_baseline_file, ensure_ascii=False, indent=2)
            temp_baseline_path = temp_baseline_file.name
        
        try:
            # use process_evaluation_data from process_eval_results.py
            domain_file = str(self.base_dir / domain_metadata_file) if domain_metadata_file else None
            detailed_df, stats_df = process_evaluation_data(
                temp_test_path, temp_baseline_path, temp_eval_path, domain_file
            )
            
            # extract domain scores from stats_df
            domain_scores = {}
            if not stats_df.empty:
                # get domain-specific scores
                domain_stats = stats_df[stats_df['group'] == 'domain']
                for _, row in domain_stats.iterrows():
                    domain = row['domain']
                    metric = row['metric']
                    
                    if metric == 'eval_weighted_score':
                        if domain not in domain_scores:
                            domain_scores[domain] = {}
                        domain_scores[domain]['mean'] = row['mean']
                        domain_scores[domain]['std'] = row['std'] 
                        domain_scores[domain]['count'] = row['count']
                
                # get overall scores for agent domain (as specified in requirements)
                overall_stats = stats_df[stats_df['group'] == 'overall']
                overall_eval_row = overall_stats[overall_stats['metric'] == 'eval_weighted_score']
                if not overall_eval_row.empty:
                    row = overall_eval_row.iloc[0]
                    domain_scores['agent'] = {
                        'mean': row['mean'],
                        'std': row['std'], 
                        'count': row['count']
                    }
            
            # ensure all three domains are present
            for domain in ["knowledge", "reasoning", "agent"]:
                if domain not in domain_scores:
                    domain_scores[domain] = {"mean": 0.0, "std": 0.0, "count": 0}
            
            return domain_scores
            
        finally:
            # cleanup temp files
            try:
                os.unlink(temp_eval_path)
                os.unlink(temp_test_path) 
                os.unlink(temp_baseline_path)
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Integrated evaluation pipeline")
    parser.add_argument("--eval_config", default="evaluation/config.yaml", help="evaluation config file path")
    parser.add_argument("--pipeline_config", default="configs/pipeline_settings.yaml", help="pipeline config file path")
    parser.add_argument("--base_dir", default=None, help="project base directory")
    parser.add_argument("--skip_eval", action="store_true", help="skip evaluation step")
    parser.add_argument("--skip_process", action="store_true", help="skip result processing step")
    parser.add_argument("--list_info", action="store_true", help="only show related information")
    parser.add_argument("--set", choices=["full", "mini"], default="full", help="dataset set to use (full or mini)")
    
    args = parser.parse_args()
    
    # initialize pipeline with dataset_set parameter
    pipeline = IntegratedEvaluationPipeline(args.eval_config, args.pipeline_config, args.base_dir, args.set)
    
    if args.list_info:
        print("📁 pipeline information:")
        print(f"  evaluation config file: {pipeline.eval_config_path}")
        print(f"  pipeline config file: {pipeline.pipeline_config_path}")
        print(f"  project directory: {pipeline.base_dir}")
        print(f"  dataset set: {pipeline.dataset_set}")
        
        model_configs = pipeline.get_model_configs()
        print(f"\n🤖 SFT model configs ({len(model_configs)}):")
        for model_name, config in model_configs.items():
            print(f"  - {model_name}: {config.get('base_model', 'N/A')}")
        
        data_models = pipeline.get_target_data_models()
        print(f"\n📊 data generation models ({len(data_models)}):")
        for model_key, config in data_models.items():
            display_name = config.get('display_name', model_key)
            print(f"  - {model_key} → {display_name}")
        
        methods = pipeline.get_evaluation_methods()
        print(f"\n⚙️ evaluation methods ({len(methods)}):")
        for method_key, config in methods.items():
            print(f"  - {method_key}: {config.get('display_name', method_key)}")
        
        dataset_files = pipeline.find_dataset_files()
        print(f"\n📊 found dataset files ({len(dataset_files)}):")
        for dataset_file in dataset_files:
            print(f"  - {dataset_file.name}")
        
        return
    
    # run pipeline
    run_eval = not args.skip_eval
    run_process = not args.skip_process
    
    # get execution parameters from config
    execution_config = pipeline.pipeline_config.get('execution', {})
    run_evaluation = execution_config.get('run_evaluation', True) and run_eval
    process_results = execution_config.get('process_results', True) and run_process
    
    success = pipeline.run_integrated_pipeline(
        run_evaluation=run_evaluation,
        process_results=process_results
    )
    
    if success:
        print("\n🎉 integrated pipeline completed successfully!")
    else:
        print("\n⚠️ integrated pipeline completed with warnings")


if __name__ == "__main__":
    main()