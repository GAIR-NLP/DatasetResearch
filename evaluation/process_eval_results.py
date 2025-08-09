"""
python calculate_scores.py \
    --test_file "$TEST_FILE" \
    --baseline_file "$BASELINE_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --verbose
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation score calculation script
function: calculate evaluation score based on test.json, baseline.json and eval.json
support different task_type corresponding to different metric weighted calculation
formula: eval_score = (eval_score - baseline) / (test - baseline)
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict


# define different task_type corresponding to metric weight configuration
TASK_TYPE_METRICS = {
    "translation": {
        "bleu": {"weight": 0.5, "metric_key1": "bleu", "metric_key2": "precisions"},
        "sacrebleu": {"weight": 0.5, "metric_key": "sacrebleu"}
    },
    "text-classification": {
        "accuracy": {"weight": 1, "metric_key": "accuracy"}
    },
    "multiple-choice": {
        "accuracy": {"weight": 1, "metric_key": "accuracy"}
    },
    "text-generation": {
        "bleu": {"weight": 0.7, "metric_key1": "bleu", "metric_key2": "precisions"},
        "sacrebleu": {"weight": 0.3, "metric_key": "sacrebleu"}
    },
    "summarization": {
        "rouge": {"weight": 1, "metric_key1": "rouge","metric_key2":"rougeL"}
    },
    "question-answering": {
        # "exact_match": {"weight": 0.1, "metric_key": "exact_match"},
        "f1": {"weight": 1, "metric_key": "f1"},
        # "bleu": {"weight": 0.2, "metric_key1": "bleu", "metric_key2": "precisions"},
        # "sacrebleu": {"weight": 0.1, "metric_key": "sacrebleu"}
    }
}

# default metric configuration (when task_type is unknown)
DEFAULT_METRICS = {
    "accuracy": {"weight": 1.0, "metric_key": "accuracy"},
    "bleu": {"weight": 1.0, "metric_key": "bleu"},
    "score": {"weight": 1.0, "metric_key": "score"}
}


def load_evaluation_data(file_path: str) -> Dict[str, Dict]:
    """
    load evaluation data file, support complex JSON format
    
    Args:
        file_path: JSON file path
    
    Returns:
        dictionary containing task_id and detailed evaluation data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        evaluation_data = {}
        
        if isinstance(data, list):
            # list format: [{"task_id": "xxx", "results": {...}}, ...]
            for item in data:
                task_id = item.get("task_id")
                if task_id is not None:
                    evaluation_data[task_id] = {
                        "task_type": item.get("task_type", "unknown"),
                        "results": item.get("results", {}),
                        "baseline": item.get("baseline", False)
                    }
        elif isinstance(data, dict):
            # dictionary format: {"task_id": {"results": {...}}, ...}
            for task_id, value in data.items():
                if isinstance(value, dict):
                    evaluation_data[task_id] = {
                        "task_type": value.get("task_type", "unknown"),
                        "results": value.get("results", {}),
                        "baseline": value.get("baseline", False)
                    }
        
        print(f"✅ successfully loaded {file_path}")
        print(f"📊 contains {len(evaluation_data)} tasks")
        return evaluation_data
    
    except Exception as e:
        print(f"❌ load file failed {file_path}: {e}")
        return {}


def extract_metric_score(results: Dict, metric_config: Dict) -> Optional[float]:
    """
    extract specified metric score from results
    
    Args:
        results: evaluation result dictionary
        metric_config: metric configuration dictionary
    
    Returns:
        extracted score, if not found, return None
    """
    # print(metric_config)
    if "metric_key1" in metric_config:
        metric_key1 =metric_config["metric_key1"]
        metric_key2 =metric_config["metric_key2"]
        if metric_key1 in results:
            score = results[metric_key1]
            if score:
                score = score[metric_key2]
                if isinstance(score, dict):
                    # if score is a dictionary, try to get the value
                    if "score" in score:
                        return float(score["score"])
                    elif "value" in score:
                        return float(score["value"])
                    else:
                        # get the first value
                        
                        for key, value in score.items():
                            if isinstance(value, (int, float)):
                                return float(value)
                elif isinstance(score, list):
                    # print(f"first score: {score}")
                    return float(score[0])
                elif isinstance(score, (int, float)):
                    return float(score)
    else:
        metric_key = metric_config["metric_key"]
        
        # try to extract score from results
        if metric_key in results:
            score = results[metric_key]
            if isinstance(score, dict):
                # if score is a dictionary, try to get the value
                if "score" in score:
                    return float(score["score"])
                elif "value" in score:
                    return float(score["value"])
                else:
                    # get the first value
                    for key, value in score.items():
                        if isinstance(value, (int, float)):
                            return float(value)
            elif isinstance(score, (int, float)):
                return float(score)
        
        return None


def calculate_weighted_score(task_type: str, results: Dict) -> Tuple[float, Dict]:
    """
    calculate weighted score based on task_type
    
    Args:
        task_type: task type
        results: evaluation result
    
    Returns:
        weighted score and detailed metric score
    """
    # 获取该task_type的metric配置
    print(results)
    metrics_config = TASK_TYPE_METRICS.get(task_type, DEFAULT_METRICS)
    
    weighted_score = 0.0
    total_weight = 0.0
    metric_scores = {}
    
    print(f"🔍 process task_type: {task_type}")
    print(f"📋 use metrics: {list(metrics_config.keys())}")
    
    for metric_name, config in metrics_config.items():
        score = extract_metric_score(results, config)
        if score is not None:
            weight = config["weight"]
            weighted_score += score * weight
            total_weight += weight
            metric_scores[metric_name] = score
            print(f"  ✅ {metric_name}: {score:.4f} (weight: {weight})")
        else:
            print(f"  ❌ {metric_name}: no score found")
    
    if total_weight > 0:
        final_score = weighted_score / total_weight
        print(f"📊 weighted average score: {final_score:.4f}")
        return final_score, metric_scores
    else:
        print(f"⚠️ no valid score found")
        return 0.0, {}


def calculate_conversion_score(result_score: float, test_score: float) -> float:
    """
    calculate conversion score
    
    Args:
        result_score: result score
        test_score: test score
    
    Returns:
        conversion score
    """
    if test_score == 0.0:
        # avoid division by zero
        return 0.0
    
    conversion_score = result_score / test_score
    return conversion_score


def load_domain_mapping(json_file: str) -> Dict[str, str]:
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
        
        print(f"✅ successfully loaded domain mapping, contains {len(domain_mapping)} task_ids")
        return domain_mapping
    except Exception as e:
        print(f"❌ load domain mapping failed: {e}")
        return {}


def process_evaluation_data(test_file: str, baseline_file: str, eval_file: str, 
                          domain_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    process evaluation data and generate results
    
    Args:
        test_file: test score file path
        baseline_file: baseline score file path
        eval_file: evaluation score file path
        domain_file: JSON file containing original_metadata (optional)
    
    Returns:
        detailed result DataFrame and statistical result DataFrame
    """
    # load all evaluation data
    test_data = load_evaluation_data(test_file)
    baseline_data = load_evaluation_data(baseline_file)
    eval_data = load_evaluation_data(eval_file)
    
    if not test_data or not baseline_data:
        print("❌ cannot load test.json or baseline.json file")
        return pd.DataFrame(), pd.DataFrame()
    
    # load domain mapping
    domain_mapping = {}
    if domain_file:
        domain_mapping = load_domain_mapping(domain_file)
    
    # convert list format to dict format, for easy lookup
    test_dict = {sample['task_id']: sample for sample in test_data} if isinstance(test_data, list) else test_data
    baseline_dict = {sample['task_id']: sample for sample in baseline_data} if isinstance(baseline_data, list) else baseline_data
    eval_dict = {sample['task_id']: sample for sample in eval_data} if isinstance(eval_data, list) else eval_data
    
    # baseline contains all task_ids
    all_task_ids = list(baseline_dict.keys())
    
    print(f"\n📋 process {len(all_task_ids)} tasks")
    
    # build detailed results
    detailed_results = []
    eval_tasks = []  # record tasks in eval
    
    for task_id in sorted(all_task_ids):
        # get task type (use task_type in eval first)
        task_type = "unknown"
        if task_id in eval_dict:
            task_type = eval_dict[task_id].get("task_type", "unknown")
        elif task_id in test_dict:
            task_type = test_dict[task_id].get("task_type", "unknown")
        elif task_id in baseline_dict:
            task_type = baseline_dict[task_id].get("task_type", "unknown")
        
        # get domain information
        domain = domain_mapping.get(task_id, "unknown")
        
        task_config = TASK_TYPE_METRICS.get(task_type, DEFAULT_METRICS)
        # get original metric score from each file
        test_metrics = {}
        baseline_metrics = {}
        eval_metrics = {}
        
        if task_id in test_dict:
            test_metrics = test_dict[task_id].get("results", {})
        if task_id in baseline_dict:
            baseline_metrics = baseline_dict[task_id].get("results", {})
        if task_id in eval_dict:
            eval_metrics = eval_dict[task_id].get("results", {})
            eval_tasks.append(task_id)
        
        # first calculate conversion score (result/test), then weighted
        baseline_conversion_scores = {}
        eval_conversion_scores = {}
        
        # use extract_metric_score to get each metric score and calculate conversion score
        for metric_name, config in task_config.items():
            weight = config["weight"]
            
            # get test score
            test_score = extract_metric_score(test_metrics, config)
            if test_score is None or test_score == 0:
                continue
                
            # get baseline score and calculate conversion score
            baseline_score = extract_metric_score(baseline_metrics, config)
            if baseline_score is not None:
                baseline_conversion_scores[metric_name] = baseline_score / test_score
            else:
                baseline_conversion_scores[metric_name] = 0.0
                
            # get eval score and calculate conversion score
            eval_score = extract_metric_score(eval_metrics, config)
            if eval_score is not None:
                eval_conversion_scores[metric_name] = eval_score / test_score
            else:
                eval_conversion_scores[metric_name] = 0.0
        
        # calculate weighted score based on task_type
        baseline_weighted_score = 0.0
        eval_weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, config in task_config.items():
            weight = config["weight"]
            
            if metric_name in baseline_conversion_scores:
                baseline_weighted_score += baseline_conversion_scores[metric_name] * weight
                total_weight += weight
            
            if metric_name in eval_conversion_scores:
                eval_weighted_score += eval_conversion_scores[metric_name] * weight
        
        if total_weight > 0:
            baseline_weighted_score /= total_weight
            eval_weighted_score /= total_weight
        
        # build result record
        result_record = {
            "task_id": task_id,
            "task_type": task_type,
            "domain": domain,
            "baseline_weighted_score": baseline_weighted_score,
            "eval_weighted_score": eval_weighted_score,
            "in_eval": task_id in eval_dict
        }
        
        # add detailed metric conversion score
        for metric_name in task_config.keys():
            result_record[f"baseline_{metric_name}_conversion"] = baseline_conversion_scores.get(metric_name, 0.0)
            result_record[f"eval_{metric_name}_conversion"] = eval_conversion_scores.get(metric_name, 0.0)
        
        detailed_results.append(result_record)
    
    detailed_df = pd.DataFrame(detailed_results)
    
    # calculate statistical results
    if eval_tasks:
        # calculate statistics by task_type and domain
        stats_results = []
        
        # extract weighted score of eval_tasks from detailed_df
        eval_baseline_weighted = []
        eval_eval_weighted = []
        
        for task_id in eval_tasks:
            task_row = detailed_df[detailed_df['task_id'] == task_id]
            if not task_row.empty:
                eval_baseline_weighted.append(task_row['baseline_weighted_score'].iloc[0])
                eval_eval_weighted.append(task_row['eval_weighted_score'].iloc[0])
        
        # overall statistics
        stats_results.extend([
            {
                "group": "overall",
                "task_type": "overall",
                "domain": "overall",
                "metric": "baseline_weighted_score",
                "mean": np.mean(eval_baseline_weighted),
                "std": np.std(eval_baseline_weighted),
                "min": np.min(eval_baseline_weighted),
                "max": np.max(eval_baseline_weighted),
                "count": len(eval_baseline_weighted)
            },
            {
                "group": "overall",
                "task_type": "overall",
                "domain": "overall",
                "metric": "eval_weighted_score",
                "mean": np.mean(eval_eval_weighted),
                "std": np.std(eval_eval_weighted),
                "min": np.min(eval_eval_weighted),
                "max": np.max(eval_eval_weighted),
                "count": len(eval_eval_weighted)
            }
        ])
        
        # calculate statistics by task_type
        task_type_groups = defaultdict(list)
        for task_id in eval_tasks:
            task_row = detailed_df[detailed_df['task_id'] == task_id]
            if not task_row.empty:
                task_type = task_row['task_type'].iloc[0]
                task_type_groups[task_type].append({
                    "baseline_weighted_score": task_row['baseline_weighted_score'].iloc[0],
                    "eval_weighted_score": task_row['eval_weighted_score'].iloc[0]
                })
        
        for task_type, scores in task_type_groups.items():
            if len(scores) > 0:
                baseline_scores = [s["baseline_weighted_score"] for s in scores]
                eval_scores = [s["eval_weighted_score"] for s in scores]
                
                stats_results.extend([
                    {
                        "group": "task_type",
                        "task_type": task_type,
                        "domain": "overall",
                        "metric": "baseline_weighted_score",
                        "mean": np.mean(baseline_scores),
                        "std": np.std(baseline_scores),
                        "min": np.min(baseline_scores),
                        "max": np.max(baseline_scores),
                        "count": len(baseline_scores)
                    },
                    {
                        "group": "task_type",
                        "task_type": task_type,
                        "domain": "overall",
                        "metric": "eval_weighted_score",
                        "mean": np.mean(eval_scores),
                        "std": np.std(eval_scores),
                        "min": np.min(eval_scores),
                        "max": np.max(eval_scores),
                        "count": len(eval_scores)
                    }
                ])
        
        # calculate statistics by domain (especially focus on knowledge and reasoning)
        # for knowledge and reasoning, include all task_ids, and set eval score to 0 if not in eval
        domain_groups = defaultdict(list)
        
        # 获取所有task_id的domain信息
        all_task_domains = {}
        for _, row in detailed_df.iterrows():
            task_id = row['task_id']
            domain = row['domain']
            all_task_domains[task_id] = domain
        
        # for knowledge and reasoning domain, include all task_ids
        for task_id, domain in all_task_domains.items():
            if domain in ['knowledge', 'reasoning']:
                task_row = detailed_df[detailed_df['task_id'] == task_id]
                if not task_row.empty:
                    baseline_score = task_row['baseline_weighted_score'].iloc[0]
                    # if task_id in eval, use actual score; otherwise set to 0
                    if task_id in eval_tasks:
                        eval_score = task_row['eval_weighted_score'].iloc[0]
                    else:
                        eval_score = 0.0
                    
                    domain_groups[domain].append({
                        "baseline_weighted_score": baseline_score,
                        "eval_weighted_score": eval_score,
                        "in_eval": task_id in eval_tasks
                    })
        
        # for other domains, only include task_ids in eval
        for task_id in eval_tasks:
            task_row = detailed_df[detailed_df['task_id'] == task_id]
            if not task_row.empty:
                domain = task_row['domain'].iloc[0]
                if domain not in ['knowledge', 'reasoning']:  # avoid duplicate addition
                    domain_groups[domain].append({
                        "baseline_weighted_score": task_row['baseline_weighted_score'].iloc[0],
                        "eval_weighted_score": task_row['eval_weighted_score'].iloc[0],
                        "in_eval": True
                    })
        
        for domain, scores in domain_groups.items():
            if len(scores) > 0:
                baseline_scores = [s["baseline_weighted_score"] for s in scores]
                eval_scores = [s["eval_weighted_score"] for s in scores]
                in_eval_count = sum(1 for s in scores if s["in_eval"])
                total_count = len(scores)
                
                stats_results.extend([
                    {
                        "group": "domain",
                        "task_type": "overall",
                        "domain": domain,
                        "metric": "baseline_weighted_score",
                        "mean": np.mean(baseline_scores),
                        "std": np.std(baseline_scores),
                        "min": np.min(baseline_scores),
                        "max": np.max(baseline_scores),
                        "count": total_count,
                        "in_eval_count": in_eval_count
                    },
                    {
                        "group": "domain",
                        "task_type": "overall",
                        "domain": domain,
                        "metric": "eval_weighted_score",
                        "mean": np.mean(eval_scores),
                        "std": np.std(eval_scores),
                        "min": np.min(eval_scores),
                        "max": np.max(eval_scores),
                        "count": total_count,
                        "in_eval_count": in_eval_count
                    }
                ])
        
        # calculate statistics by task_type and domain
        # for knowledge and reasoning, include all task_ids, and set eval score to 0 if not in eval
        cross_groups = defaultdict(lambda: defaultdict(list))
        
        # for knowledge and reasoning domain, include all task_ids
        for task_id, domain in all_task_domains.items():
            if domain in ['knowledge', 'reasoning']:
                task_row = detailed_df[detailed_df['task_id'] == task_id]
                if not task_row.empty:
                    task_type = task_row['task_type'].iloc[0]
                    baseline_score = task_row['baseline_weighted_score'].iloc[0]
                    # if task_id in eval, use actual score; otherwise set to 0
                    if task_id in eval_tasks:
                        eval_score = task_row['eval_weighted_score'].iloc[0]
                    else:
                        eval_score = 0.0
                    
                    cross_groups[task_type][domain].append({
                        "baseline_weighted_score": baseline_score,
                        "eval_weighted_score": eval_score,
                        "in_eval": task_id in eval_tasks
                    })
        
        # for other domains, only include task_ids in eval
        for task_id in eval_tasks:
            task_row = detailed_df[detailed_df['task_id'] == task_id]
            if not task_row.empty:
                task_type = task_row['task_type'].iloc[0]
                domain = task_row['domain'].iloc[0]
                if domain not in ['knowledge', 'reasoning']:  # avoid duplicate addition
                    cross_groups[task_type][domain].append({
                        "baseline_weighted_score": task_row['baseline_weighted_score'].iloc[0],
                        "eval_weighted_score": task_row['eval_weighted_score'].iloc[0],
                        "in_eval": True
                    })
        
        for task_type, domains in cross_groups.items():
            for domain, scores in domains.items():
                if len(scores) > 0:
                    baseline_scores = [s["baseline_weighted_score"] for s in scores]
                    eval_scores = [s["eval_weighted_score"] for s in scores]
                    in_eval_count = sum(1 for s in scores if s["in_eval"])
                    total_count = len(scores)
                    
                    stats_results.extend([
                        {
                            "group": "cross",
                            "task_type": task_type,
                            "domain": domain,
                            "metric": "baseline_weighted_score",
                            "mean": np.mean(baseline_scores),
                            "std": np.std(baseline_scores),
                            "min": np.min(baseline_scores),
                            "max": np.max(baseline_scores),
                            "count": total_count,
                            "in_eval_count": in_eval_count
                        },
                        {
                            "group": "cross",
                            "task_type": task_type,
                            "domain": domain,
                            "metric": "eval_weighted_score",
                            "mean": np.mean(eval_scores),
                            "std": np.std(eval_scores),
                            "min": np.min(eval_scores),
                            "max": np.max(eval_scores),
                            "count": total_count,
                            "in_eval_count": in_eval_count
                        }
                    ])
        
        stats_df = pd.DataFrame(stats_results)
    else:
        stats_df = pd.DataFrame()
    
    return detailed_df, stats_df


def save_results(detailed_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str):
    """
    save results to CSV file
    
    Args:
        detailed_df: detailed result DataFrame
        stats_df: statistical result DataFrame
        output_dir: output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # save detailed results
    detailed_file = output_path / "detailed_scores.csv"
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8')
    print(f"✅ detailed results saved to: {detailed_file}")
    
    # save statistical results
    if not stats_df.empty:
        stats_file = output_path / "score_statistics.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8')
        print(f"✅ statistical results saved to: {stats_file}")
        
        # print statistical information
        print("\n📈 statistical information:")
        print("="*80)
        
        # overall statistics
        overall_stats = stats_df[stats_df['group'] == 'overall']
        if not overall_stats.empty:
            print("\n🌐 overall statistics:")
            print("-" * 60)
            for _, row in overall_stats.iterrows():
                print(f"{row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                      f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
        
        # display by domain (especially focus on knowledge and reasoning)
        domain_stats = stats_df[stats_df['group'] == 'domain']
        if not domain_stats.empty:
            print("\n🎯 domain statistics:")
            print("-" * 60)
            for domain in sorted(domain_stats['domain'].unique()):
                domain_data = domain_stats[domain_stats['domain'] == domain]
                print(f"\n🔹 {domain.upper()}:")
                for _, row in domain_data.iterrows():
                    if 'in_eval_count' in row:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']}, eval中: {row['in_eval_count']})")
                    else:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
        
        # display by task_type
        task_type_stats = stats_df[stats_df['group'] == 'task_type']
        if not task_type_stats.empty:
            print("\n📋 task type statistics:")
            print("-" * 60)
            for task_type in sorted(task_type_stats['task_type'].unique()):
                task_data = task_type_stats[task_type_stats['task_type'] == task_type]
                print(f"\n🔹 {task_type.upper()}:")
                for _, row in task_data.iterrows():
                    print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                          f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
        
        # cross group statistics (task_type + domain)
        cross_stats = stats_df[stats_df['group'] == 'cross']
        if not cross_stats.empty:
            print("\n🔀 cross group statistics (task type × domain):")
            print("-" * 60)
            for task_type in sorted(cross_stats['task_type'].unique()):
                task_data = cross_stats[cross_stats['task_type'] == task_type]
                print(f"\n🔹 {task_type.upper()}:")
                for domain in sorted(task_data['domain'].unique()):
                    domain_data = task_data[task_data['domain'] == domain]
                    print(f"  📍 {domain}:")
                    for _, row in domain_data.iterrows():
                        if 'in_eval_count' in row:
                            print(f"    {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                                  f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']}, eval中: {row['in_eval_count']})")
                        else:
                            print(f"    {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                                  f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
        
        # especially focus on knowledge and reasoning
        knowledge_stats = domain_stats[domain_stats['domain'] == 'knowledge']
        reasoning_stats = domain_stats[domain_stats['domain'] == 'reasoning']
        
        if not knowledge_stats.empty or not reasoning_stats.empty:
            print("\n🎯 knowledge vs reasoning comparison:")
            print("=" * 60)
            
            if not knowledge_stats.empty:
                print(f"\n📚 knowledge domain:")
                for _, row in knowledge_stats.iterrows():
                    if 'in_eval_count' in row:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']}, eval中: {row['in_eval_count']})")
                    else:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
            
            if not reasoning_stats.empty:
                print(f"\n🧠 reasoning domain:")
                for _, row in reasoning_stats.iterrows():
                    if 'in_eval_count' in row:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']}, eval中: {row['in_eval_count']})")
                    else:
                        print(f"  {row['metric']:>15}: {row['mean']:.4f} ± {row['std']:.4f} "
                              f"[{row['min']:.4f}, {row['max']:.4f}] (n={row['count']})")
    else:
        print("⚠️ no tasks in eval, no statistical results generated")


def main():
    parser = argparse.ArgumentParser(description="evaluation score calculation tool - support multiple metrics and domain analysis")
    parser.add_argument("--test_file", required=True, help="test score file path (test.json)")
    parser.add_argument("--baseline_file", required=True, help="baseline score file path (baseline.json)")
    parser.add_argument("--eval_file", required=True, help="evaluation score file path (eval.json)")
    parser.add_argument("--domain_file", help="JSON file path containing original_metadata (for domain analysis)")
    parser.add_argument("--output_dir", default="./score_results", help="output directory")
    parser.add_argument("--verbose", action="store_true", help="display detailed information")
    
    args = parser.parse_args()
    
    print("🔄 evaluation score calculation tool - multiple metrics and domain analysis")
    print("="*80)
    print(f"test file: {args.test_file}")
    print(f"baseline file: {args.baseline_file}")
    print(f"evaluation file: {args.eval_file}")
    if args.domain_file:
        print(f"domain file: {args.domain_file}")
    else:
        print(f"domain file: not provided (use default domain='unknown')")
    print(f"output directory: {args.output_dir}")
    print("="*80)
    
    # check if files exist
    required_files = [args.test_file, args.baseline_file, args.eval_file]
    if args.domain_file:
        required_files.append(args.domain_file)
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ file not found: {file_path}")
            return
    
    # process evaluation data
    detailed_df, stats_df = process_evaluation_data(
        args.test_file, args.baseline_file, args.eval_file, args.domain_file
    )
    
    if detailed_df.empty:
        print("❌ process failed, program exit")
        return
    
    # display processing results
    print(f"\n📊 processing results:")
    print(f"  total tasks: {len(detailed_df)}")
    print(f"  tasks in eval: {len(detailed_df[detailed_df['in_eval'] == True])}")
    print(f"  tasks not in eval: {len(detailed_df[detailed_df['in_eval'] == False])}")
    
    # display task_type distribution
    task_type_counts = detailed_df['task_type'].value_counts()
    print(f"\n📋 task type distribution:")
    for task_type, count in task_type_counts.items():
        print(f"  {task_type}: {count}")
    
    # display domain distribution
    domain_counts = detailed_df['domain'].value_counts()
    print(f"\n🌐 domain distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count}")
    
    if args.verbose:
        print(f"\n📋 first 10 tasks details:")
        print(detailed_df.head(10).to_string(index=False))
    
    # save results
    save_results(detailed_df, stats_df, args.output_dir)
    
    print(f"\n✅ processing completed! results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()