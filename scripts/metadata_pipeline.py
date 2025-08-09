#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的元数据生成与评估管道
整合generate_generation_set_metadata和evaluate_metadata功能
"""
import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import logging
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

# 导入重构后的模块
from scripts.method.generate_generation_set_metadata import MetadataGenerator, MetadataGenerationConfig
from evaluation.evaluate_metadata import MetadataEvaluator, EvaluationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """管道配置"""
    # 管道控制配置
    pipeline: Dict[str, Any] = field(default_factory=lambda: {
        "steps": {
            "generate_metadata": True,
            "evaluate_metadata": True
        },
        "mode": "full"  # full | generate_only | evaluate_only
    })
    
    # 元数据生成配置
    metadata_generation: Dict[str, Any] = field(default_factory=dict)
    
    # 输入文件配置
    input_files: Dict[str, Any] = field(default_factory=dict)
    
    # 批量评估配置
    batch_evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # 输出路径配置
    output_paths: Dict[str, str] = field(default_factory=dict)
    
    # 评估参数配置
    evaluation_params: Dict[str, Any] = field(default_factory=dict)
    
    # LLM配置
    llm_config: Dict[str, Any] = field(default_factory=dict)
    
    # 提示词配置
    prompts: Dict[str, str] = field(default_factory=dict)
    
    # 统计配置
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """从YAML配置文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            pipeline=config_data.get('pipeline', cls().pipeline),
            metadata_generation=config_data.get('metadata_generation', {}),
            input_files=config_data.get('input_files', {}),
            batch_evaluation=config_data.get('batch_evaluation', {}),
            output_paths=config_data.get('output_paths', {}),
            evaluation_params=config_data.get('evaluation_params', {}),
            llm_config=config_data.get('llm_config', {}),
            prompts=config_data.get('prompts', {}),
            statistics=config_data.get('statistics', {})
        )
    
    def to_generation_config(self) -> MetadataGenerationConfig:
        """转换为元数据生成配置"""
        return MetadataGenerationConfig(
            input_data_template=self.metadata_generation.get('input_data_template', 
                "LLaMA-Factory/data/deep_research_dataset/{model}/{dataset_id}.json"),
            samples_count=self.metadata_generation.get('samples_count', 5),
            models=self.metadata_generation.get('models', ["o3-w", "o3-wo", "gemini", "grok", "openai"]),
            output_paths=self.metadata_generation.get('output_paths', {}),
            generation_params=self.metadata_generation.get('generation_params', {}),
            llm_config=self.llm_config,
            prompts=self.prompts
        )
    
    def to_evaluation_config(self) -> EvaluationConfig:
        """转换为评估配置"""
        return EvaluationConfig(
            input_files=self.input_files,
            batch_evaluation=self.batch_evaluation,
            output_paths=self.output_paths,
            evaluation_params=self.evaluation_params,
            llm_config=self.llm_config,
            prompts=self.prompts,
            statistics=self.statistics
        )
    
    def override_from_env(self):
        """从环境变量覆盖配置"""
        if os.getenv('LLM_API_KEY'):
            self.llm_config['api_key'] = os.getenv('LLM_API_KEY')
        if os.getenv('LLM_API_BASE'):
            self.llm_config['api_base'] = os.getenv('LLM_API_BASE')


class MetadataPipeline:
    """统一的元数据生成与评估管道"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.generation_config = config.to_generation_config()
        self.evaluation_config = config.to_evaluation_config()
        
        # 初始化组件
        self.metadata_generator = None
        self.metadata_evaluator = None
        
        logger.info("初始化元数据管道")
    
    def run_full_pipeline(self, input_file: str, base_output_dir: str = None):
        """运行完整管道"""
        logger.info("开始运行完整元数据管道")
        
        # 确定输出目录
        if not base_output_dir:
            base_output_dir = self.config.output_paths.get('output_dir', './evaluation/results/metadata_evaluation/')
        
        # 步骤1: 生成元数据
        if self.config.pipeline['steps'].get('generate_metadata', True):
            logger.info("步骤1: 开始生成元数据")
            self._run_metadata_generation(input_file)
            logger.info("步骤1: 元数据生成完成")
        else:
            logger.info("跳过元数据生成步骤")
        
        # 步骤2: 评估元数据
        if self.config.pipeline['steps'].get('evaluate_metadata', True):
            logger.info("步骤2: 开始评估元数据")
            self._run_metadata_evaluation(base_output_dir)
            logger.info("步骤2: 元数据评估完成")
        else:
            logger.info("跳过元数据评估步骤")
        
        logger.info("完整元数据管道执行完成")
    
    def _run_metadata_generation(self, input_file: str):
        """运行元数据生成"""
        if not self.metadata_generator:
            self.metadata_generator = MetadataGenerator(self.generation_config)
        
        models = self.generation_config.models
        output_template = self.generation_config.output_paths.get(
            'generation_metadata_template', 
            'datasets/results/generation_metadata_{model}.json'
        )
        
        # 为每个模型生成元数据
        for model in models:
            output_file = output_template.format(model=model)
            logger.info(f"为模型 {model} 生成元数据，输出到: {output_file}")
            
            try:
                self.metadata_generator.generate_metadata_for_model(
                    input_file=input_file,
                    output_file=output_file,
                    model=model
                )
                logger.info(f"模型 {model} 的元数据生成完成")
            except Exception as e:
                logger.error(f"模型 {model} 的元数据生成失败: {e}")
    
    def _run_metadata_evaluation(self, base_output_dir: str):
        """运行元数据评估"""
        if not self.metadata_evaluator:
            self.metadata_evaluator = MetadataEvaluator(self.evaluation_config)
        
        # 检查是否启用批量评估
        if self.config.batch_evaluation.get('enabled', False):
            logger.info("运行批量评估")
            self._run_batch_evaluation(base_output_dir)
        else:
            logger.info("运行单个文件评估")
            self._run_single_evaluation(base_output_dir)
    
    def _run_batch_evaluation(self, base_output_dir: str):
        """运行批量评估"""
        comparison_groups = self.config.batch_evaluation.get('comparison_groups', {})
        data_sources = self.config.input_files.get('data_sources', {})
        
        # 合并搜索和生成数据源
        all_sources = {}
        if 'search_datasets' in data_sources:
            all_sources.update(data_sources['search_datasets'])
        if 'generation_datasets' in data_sources:
            all_sources.update(data_sources['generation_datasets'])
        
        results = []
        
        # 处理所有比较组
        for group_name, comparisons in comparison_groups.items():
            logger.info(f"处理比较组: {group_name}")
            
            for comparison in comparisons:
                source_name = comparison.get('source')
                target_names = comparison.get('targets', [])
                
                if source_name not in all_sources:
                    logger.warning(f"源数据集 {source_name} 未找到，跳过")
                    continue
                
                source_file = all_sources[source_name]
                
                # 对每个目标进行评估
                for target_name in target_names:
                    if target_name not in all_sources:
                        logger.warning(f"目标数据集 {target_name} 未找到，跳过")
                        continue
                    
                    target_file = all_sources[target_name]
                    
                    logger.info(f"评估: {source_name} vs {target_name}")
                    
                    try:
                        # 运行单次评估
                        result = self.metadata_evaluator.evaluate_metadata_files(
                            source_file, target_file
                        )
                        
                        # 保存结果
                        output_file = os.path.join(
                            base_output_dir, 
                            f"evaluation_{source_name}_vs_{target_name}.json"
                        )
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        results.append({
                            "comparison": f"{source_name}_vs_{target_name}",
                            "source": source_name,
                            "target": target_name,
                            "result_file": output_file,
                            "statistics": result.get('statistics', {})
                        })
                        
                        logger.info(f"评估 {source_name} vs {target_name} 完成")
                        
                    except Exception as e:
                        logger.error(f"评估 {source_name} vs {target_name} 失败: {e}")
        
        # 保存综合结果
        final_results_file = os.path.join(
            base_output_dir,
            self.config.output_paths.get('final_results_filename', 'final_evaluation_results.json')
        )
        
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量评估完成，结果保存到: {final_results_file}")
    
    def _run_single_evaluation(self, base_output_dir: str):
        """运行单个文件评估"""
        json_file_path = self.config.input_files.get('json_file_path')
        
        if not json_file_path or not os.path.exists(json_file_path):
            logger.error(f"输入文件不存在: {json_file_path}")
            return
        
        logger.info(f"评估文件: {json_file_path}")
        
        try:
            result = self.metadata_evaluator.evaluate_metadata_file(json_file_path)
            
            # 保存结果
            output_file = os.path.join(base_output_dir, "metadata_evaluation_result.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"单文件评估完成，结果保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"单文件评估失败: {e}")
    
    def run_generation_only(self, input_file: str):
        """只运行元数据生成"""
        logger.info("运行元数据生成（仅生成）")
        self._run_metadata_generation(input_file)
        logger.info("元数据生成完成")
    
    def run_evaluation_only(self, base_output_dir: str = None):
        """只运行元数据评估"""
        logger.info("运行元数据评估（仅评估）")
        if not base_output_dir:
            base_output_dir = self.config.output_paths.get('output_dir', './evaluation/results/metadata_evaluation/')
        self._run_metadata_evaluation(base_output_dir)
        logger.info("元数据评估完成")


def main():
    parser = argparse.ArgumentParser(description='统一的元数据生成与评估管道')
    parser.add_argument('--config', default='configs/evaluate_metadata_config.yaml', help='配置文件路径')
    parser.add_argument('--mode', choices=['full', 'generate_only', 'evaluate_only'], 
                        default='full', help='运行模式')
    parser.add_argument('--input-file', help='输入文件（生成步骤需要）')
    parser.add_argument('--output-dir', help='输出目录（覆盖配置文件）')
    
    # LLM配置覆盖参数
    parser.add_argument('--api-key', help='覆盖API密钥')
    parser.add_argument('--api-base', help='覆盖API基础URL')
    parser.add_argument('--api-model', help='覆盖API模型')
    
    # 管道步骤控制
    parser.add_argument('--skip-generation', action='store_true', help='跳过生成步骤')
    parser.add_argument('--skip-evaluation', action='store_true', help='跳过评估步骤')
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = PipelineConfig.from_yaml(args.config)
        config.override_from_env()
        
        # 命令行参数覆盖
        if args.api_key:
            config.llm_config['api_key'] = args.api_key
        if args.api_base:
            config.llm_config['api_base'] = args.api_base
        if args.api_model:
            config.llm_config['api_model'] = args.api_model
        
        # 更新管道步骤
        if args.skip_generation:
            config.pipeline['steps']['generate_metadata'] = False
        if args.skip_evaluation:
            config.pipeline['steps']['evaluate_metadata'] = False
        
        # 覆盖模式
        if args.mode != 'full':
            config.pipeline['mode'] = args.mode
        
        # 创建管道
        pipeline = MetadataPipeline(config)
        
        # 根据模式运行
        if args.mode == 'generate_only' or config.pipeline['mode'] == 'generate_only':
            if not args.input_file:
                logger.error("生成模式需要指定 --input-file 参数")
                sys.exit(1)
            pipeline.run_generation_only(args.input_file)
        
        elif args.mode == 'evaluate_only' or config.pipeline['mode'] == 'evaluate_only':
            pipeline.run_evaluation_only(args.output_dir)
        
        else:
            # 完整管道
            if not args.input_file:
                logger.error("完整管道模式需要指定 --input-file 参数")
                sys.exit(1)
            pipeline.run_full_pipeline(args.input_file, args.output_dir)
        
        logger.info("管道执行完成")
        
    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()