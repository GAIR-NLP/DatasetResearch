# Universal Evaluation Interface Usage Guide

## Overview

The `QuestionAnsweringEvaluator` class provides a universal evaluation interface that supports:

1. **Dedicated Evaluation Script Invocation** - Directly call specialized evaluation scripts under LLaMA-Factory/evaluation
2. **Hugging Face Metrics Calculation** - Use HF evaluate library to calculate common metrics
3. **Automatic Result Parsing** - Intelligently parse evaluation script output results

## Basic Usage

### 1. Initialize Evaluator

```python
from custom_evaluation import QuestionAnsweringEvaluator

pipeline = QuestionAnsweringEvaluator(
    workspace_dir="/path/to/workspace",
    llamafactory_dir="/path/to/LLaMA-Factory"
)
```

### 2. Using Dedicated Evaluation Scripts

#### TriviaQA Evaluation
```python
eval_results = pipeline.evaluation(
    prediction_file="predictions.jsonl",
    ground_truth_file="triviaqa_test.json",
    evaluation_script="triviaqa/triviaqa.py",
    script_args={"mute": False}
)
```

#### BigCodeBench Evaluation
```python
eval_results = pipeline.evaluation(
    prediction_file="code_predictions.jsonl", 
    ground_truth_file="bigcodebench_test.json",
    evaluation_script="bigcodebench/bigcodebench.py",
    script_args={
        "timeout": 10.0,
        "parallel": 4,
        "pass_k": "1,5,10",
        "calibrated": True
    }
)
```

### 3. Using Hugging Face Metrics

```python
# Use default metrics (exact_match, f1)
eval_results = pipeline.evaluation(
    prediction_file="predictions.jsonl",
    ground_truth_file="ground_truth.json"
)

# Specify multiple metrics
eval_results = pipeline.evaluation(
    prediction_file="predictions.jsonl",
    ground_truth_file="ground_truth.json",
    metrics=['exact_match', 'f1', 'bleu', 'rouge']
)
```

## Feature Details

### Supported File Formats

#### Prediction File (JSONL format)
```json
{"predict": "answer1"}
{"predict": "answer2"}
```

Or:
```json
{"prediction": "answer1", "id": "q1"}
{"prediction": "answer2", "id": "q2"}
```

#### Ground Truth File (JSON/JSONL format)
```json
[
    {"output": "correct_answer1"},
    {"output": "correct_answer2"}
]
```

### View Available Evaluation Scripts

```python
# List all available evaluation scripts
available_scripts = pipeline.list_available_evaluation_scripts()
print(available_scripts)
# Output: {'triviaqa': ['triviaqa/triviaqa.py'], 'bigcodebench': ['bigcodebench/bigcodebench.py']}

# View script usage help
usage_info = pipeline.get_script_usage_info("triviaqa/triviaqa.py")
print(usage_info)
```

### Evaluation Result Format

Dedicated script evaluation results:
```python
{
    'exact_match': 0.8542,
    'f1': 0.8923,
    'total_samples': 1000,
    'evaluation_method': 'triviaqa/triviaqa.py',
    'timestamp': '2024-01-01 12:00:00'
}
```

BigCodeBench evaluation results:
```python
{
    'pass@1': 0.3456,
    'pass@5': 0.5234,
    'pass@10': 0.6123,
    'accuracy': 0.3456,
    'total_samples': 500,
    'evaluation_method': 'bigcodebench/bigcodebench.py'
}
```

## Adding New Evaluation Scripts

### Script Requirements

1. Support command line arguments:
   - `--prediction_file`: Path to prediction file
   - `--dataset_file`: Path to dataset file

2. Output format requirements:
   - Metric names and values separated by colon: `Metric Name: 0.1234`
   - Support common metrics: exact_match, f1, accuracy, etc.
   - Support pass@k format: `pass@1: 0.1234`

### Example Script Structure

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', required=True)
    parser.add_argument('--dataset_file', required=True)
    parser.add_argument('--custom_param', default='default_value')
    args = parser.parse_args()
    
    # Load data and evaluate
    results = evaluate_function(args.prediction_file, args.dataset_file)
    
    # Output results (standard format)
    print(f"Exact Match: {results['exact_match']}")
    print(f"F1 Score: {results['f1']}")
    print(f"Total Questions: {results['total']}")

if __name__ == '__main__':
    main()
```

## Important Notes

1. **File Path Resolution**: Supports absolute paths, relative paths, and dataset names from dataset_info.json
2. **Error Handling**: Throws detailed error messages when script execution fails
3. **Timeout Mechanism**: Evaluation script execution timeout is 10 minutes
4. **Result Saving**: Evaluation results are automatically saved to specified output directory
5. **Logging**: Detailed execution logs for debugging convenience

## Practical Usage Examples

```python
# Complete evaluation workflow example
pipeline = QuestionAnsweringEvaluator(
    workspace_dir="./workspace",
    llamafactory_dir="./LLaMA-Factory"
)

# 1. TriviaQA Evaluation
trivia_results = pipeline.evaluation(
    prediction_file="results/triviaqa_predictions.jsonl",
    ground_truth_file="triviaqa_test",  # Look up from dataset_info.json
    evaluation_script="triviaqa/triviaqa.py",
    output_dir="./eval_results"
)

# 2. Code Generation Evaluation
code_results = pipeline.evaluation(
    prediction_file="results/code_predictions.jsonl", 
    ground_truth_file="bigcodebench_test.json",
    evaluation_script="bigcodebench/bigcodebench.py",
    script_args={"timeout": 15.0, "parallel": 8},
    output_dir="./eval_results"
)

print(f"TriviaQA EM: {trivia_results['exact_match']:.4f}")
print(f"Code Pass@1: {code_results['pass@1']:.4f}")
```
```
