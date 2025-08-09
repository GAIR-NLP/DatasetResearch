#!/usr/bin/env python3
"""
Response Generation Evaluation Script
response generation evaluation script, support ROUGE, BLEU, Exact Match, etc.
"""

import json
import argparse
import re
import string
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from collections import Counter

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import rouge
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("Rouge package not available. Install with: pip install rouge")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK package not available. Install with: pip install nltk")

class ResponseEvaluator:
    """response generation evaluation tool"""
    
    def __init__(self):
        self.rouge_evaluator = Rouge() if ROUGE_AVAILABLE else None
    
    def normalize_text(self, text: str) -> str:
        """text normalization"""
        if not text:
            return ""
        
        # convert to lowercase
        text = text.lower()
        
        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_answer_aliases(self, answer: str) -> Set[str]:
        """extract multiple expressions of the answer"""
        aliases = set()
        
        if not answer:
            return aliases
        
        # original answer
        aliases.add(answer)
        aliases.add(answer.lower())
        aliases.add(self.normalize_text(answer))
        
        # process bracket content - extract content outside and inside the bracket
        # e.g. "United States (US)" -> {"United States", "US", "United States (US)"}
        bracket_pattern = r'(.+?)\s*\((.+?)\)'
        match = re.search(bracket_pattern, answer)
        if match:
            main_part = match.group(1).strip()
            bracket_part = match.group(2).strip()
            aliases.add(main_part)
            aliases.add(bracket_part)
            aliases.add(self.normalize_text(main_part))
            aliases.add(self.normalize_text(bracket_part))
        
        # process common separators
        for sep in [',', ';', '/', '|', ' or ', ' and ']:
            if sep in answer:
                parts = answer.split(sep)
                for part in parts:
                    part = part.strip()
                    if part:
                        aliases.add(part)
                        aliases.add(self.normalize_text(part))
        
        # remove empty strings
        aliases.discard("")
        
        return aliases
    
    def exact_match(self, prediction: str, ground_truth: str) -> bool:
        """exact match"""
        if not prediction or not ground_truth:
            return False
        
        pred_aliases = self.extract_answer_aliases(prediction)
        gt_aliases = self.extract_answer_aliases(ground_truth)
        
        # check if any alias matches
        return bool(pred_aliases & gt_aliases)
    
    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """calculate F1 score (based on word level)"""
        if not prediction or not ground_truth:
            return 0.0
        
        pred_tokens = set(self.normalize_text(prediction).split())
        gt_tokens = set(self.normalize_text(ground_truth).split())
        
        if not pred_tokens and not gt_tokens:
            return 1.0
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common_tokens = pred_tokens & gt_tokens
        
        if not common_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def calculate_rouge(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """calculate ROUGE score"""
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE not available, skip ROUGE calculation")
            return {}
        
        if not predictions or not ground_truths:
            return {}
        
        # filter out empty values
        valid_pairs = []
        for pred, gt in zip(predictions, ground_truths):
            if pred and gt and pred.strip() and gt.strip():
                valid_pairs.append((pred.strip(), gt.strip()))
        
        if not valid_pairs:
            logger.warning("no valid prediction-ground truth pairs")
            return {}
        
        try:
            pred_list = [pair[0] for pair in valid_pairs]
            gt_list = [pair[1] for pair in valid_pairs]
            
            scores = self.rouge_evaluator.get_scores(pred_list, gt_list, avg=True)
            
            return {
                'rouge-1-f': scores['rouge-1']['f'],
                'rouge-1-p': scores['rouge-1']['p'],
                'rouge-1-r': scores['rouge-1']['r'],
                'rouge-2-f': scores['rouge-2']['f'],
                'rouge-2-p': scores['rouge-2']['p'],
                'rouge-2-r': scores['rouge-2']['r'],
                'rouge-l-f': scores['rouge-l']['f'],
                'rouge-l-p': scores['rouge-l']['p'],
                'rouge-l-r': scores['rouge-l']['r'],
            }
        except Exception as e:
            logger.error(f"ROUGE calculation failed: {e}")
            return {}
    
    def calculate_bleu(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """calculate BLEU score"""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available, skip BLEU calculation")
            return {}
        
        if not predictions or not ground_truths:
            return {}
        
        # filter out empty values and tokenize
        valid_pairs = []
        for pred, gt in zip(predictions, ground_truths):
            if pred and gt and pred.strip() and gt.strip():
                try:
                    pred_tokens = word_tokenize(pred.strip().lower())
                    gt_tokens = word_tokenize(gt.strip().lower())
                    if pred_tokens and gt_tokens:
                        valid_pairs.append((pred_tokens, [gt_tokens]))  # BLEU requires reference answer to be a list of lists
                except Exception as e:
                    logger.warning(f"tokenization failed: {e}")
                    continue
        
        if not valid_pairs:
            logger.warning("no valid prediction-ground truth pairs")
            return {}
        
        try:
            pred_list = [pair[0] for pair in valid_pairs]
            ref_list = [pair[1] for pair in valid_pairs]
            
            # calculate corpus-level BLEU
            bleu_scores = {}
            
            # BLEU-1 to BLEU-4
            for n in range(1, 5):
                weights = [1.0/n] * n + [0.0] * (4-n)
                try:
                    score = corpus_bleu(ref_list, pred_list, weights=weights)
                    bleu_scores[f'bleu-{n}'] = score
                except Exception as e:
                    logger.warning(f"BLEU-{n} calculation failed: {e}")
                    bleu_scores[f'bleu-{n}'] = 0.0
            
            return bleu_scores
        except Exception as e:
            logger.error(f"BLEU calculation failed: {e}")
            return {}
    
    def load_predictions(self, prediction_file: str) -> List[str]:
        """load prediction results"""
        predictions = []
        
        try:
            with open(prediction_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # parse JSONL format
                        data = json.loads(line)
                        
                        # extract prediction text
                        if 'predict' in data:
                            response = data['predict']
                        elif 'prediction' in data:
                            response = data['prediction']
                        elif 'output' in data:
                            response = data['output']
                        elif 'answer' in data:
                            response = data['answer']
                        elif 'response' in data:
                            response = data['response']
                        else:
                            logger.warning(f"line {line_idx + 1}: no prediction field found")
                            predictions.append("")
                            continue
                        
                        predictions.append(str(response) if response else "")
                        
                    except json.JSONDecodeError:
                        # try as plain text
                        predictions.append(line)
        
        except Exception as e:
            logger.error(f"load prediction file failed: {e}")
            return []
        
        logger.info(f"loaded {len(predictions)} prediction results")
        return predictions
    
    def load_ground_truth(self, ground_truth_file: str) -> List[str]:
        """load ground truth"""
        ground_truths = []
        
        if not ground_truth_file or not Path(ground_truth_file).exists():
            logger.warning("no ground truth file provided or file not found")
            return []
        
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # parse JSONL format
                        data = json.loads(line)
                        
                        # extract ground truth
                        if 'answer' in data:
                            answer = data['answer']
                        elif 'ground_truth' in data:
                            answer = data['ground_truth']
                        elif 'label' in data:
                            answer = data['label']
                        elif 'target' in data:
                            answer = data['target']
                        else:
                            logger.warning(f"line {line_idx + 1}: no answer field found")
                            ground_truths.append("")
                            continue
                        
                        ground_truths.append(str(answer) if answer else "")
                        
                    except json.JSONDecodeError:
                        # try as plain text
                        ground_truths.append(line)
        
        except Exception as e:
            logger.error(f"load ground truth file failed: {e}")
            return []
        
        logger.info(f"loaded {len(ground_truths)} ground truth")
        return ground_truths
    
    def evaluate(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, Any]:
        """comprehensive evaluation"""
        if not predictions:
            return {"error": "No predictions available"}
        
        if not ground_truths:
            logger.warning("no ground truth, only perform basic statistics")
            return {
                "total_predictions": len(predictions),
                "empty_predictions": sum(1 for p in predictions if not p.strip()),
                "avg_length": sum(len(p.split()) for p in predictions) / len(predictions) if predictions else 0
            }
        
        # ensure length consistency
        min_length = min(len(predictions), len(ground_truths))
        if len(predictions) != len(ground_truths):
            logger.warning(f"length mismatch - prediction: {len(predictions)}, ground truth: {len(ground_truths)}")
            logger.warning(f"use minimum length: {min_length}")
        
        predictions = predictions[:min_length]
        ground_truths = ground_truths[:min_length]
        
        # calculate basic metrics
        exact_matches = []
        f1_scores = []
        
        for pred, gt in zip(predictions, ground_truths):
            exact_matches.append(self.exact_match(pred, gt))
            f1_scores.append(self.f1_score(pred, gt))
        
        results = {
            "exact_match": sum(exact_matches) / len(exact_matches) if exact_matches else 0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "total": len(predictions),
            "exact_match_count": sum(exact_matches),
        }
        
        # calculate ROUGE score
        rouge_scores = self.calculate_rouge(predictions, ground_truths)
        results.update(rouge_scores)
        
        # calculate BLEU score
        bleu_scores = self.calculate_bleu(predictions, ground_truths)
        results.update(bleu_scores)
        
        # statistics
        results.update({
            "avg_pred_length": sum(len(p.split()) for p in predictions) / len(predictions) if predictions else 0,
            "avg_gt_length": sum(len(gt.split()) for gt in ground_truths) / len(ground_truths) if ground_truths else 0,
            "empty_predictions": sum(1 for p in predictions if not p.strip()),
            "empty_ground_truths": sum(1 for gt in ground_truths if not gt.strip()),
        })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Response Generation Evaluation Script")
    parser.add_argument("--prediction_file", required=True, help="prediction result file (JSONL format)")
    parser.add_argument("--ground_truth_file", help="ground truth file (JSONL format)")
    parser.add_argument("--output_dir", help="output directory")
    parser.add_argument("--detailed_analysis", action="store_true", help="detailed analysis")
    
    args = parser.parse_args()
    
    evaluator = ResponseEvaluator()
    
    # load prediction results
    logger.info("loading prediction results...")
    predictions = evaluator.load_predictions(args.prediction_file)
    
    if not predictions:
        logger.error("no valid prediction results")
        return
    
    # load ground truth
    ground_truths = []
    if args.ground_truth_file:
        logger.info("loading ground truth...")
        ground_truths = evaluator.load_ground_truth(args.ground_truth_file)
    
    # evaluate
    logger.info("evaluating...")
    results = evaluator.evaluate(predictions, ground_truths)
    
    # print results
    print("\n" + "="*60)
    print("Response Generation evaluation results")
    print("="*60)
    
    if "error" in results:
        print(f"error: {results['error']}")
    else:
        # basic metrics
        if "exact_match" in results:
            print(f"Exact Match: {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
            print(f"F1 Score: {results['f1']:.4f}")
            print(f"exact match count: {results.get('exact_match_count', 0)}")
        
        print(f"total samples: {results.get('total', len(predictions))}")
        print(f"empty predictions: {results.get('empty_predictions', 0)}")
        
        # ROUGE score
        rouge_keys = [k for k in results.keys() if k.startswith('rouge')]
        if rouge_keys:
            print(f"\nROUGE score:")
            for key in sorted(rouge_keys):
                print(f"  {key}: {results[key]:.4f}")
        
        # BLEU score
        bleu_keys = [k for k in results.keys() if k.startswith('bleu')]
        if bleu_keys:
            print(f"\nBLEU score:")
            for key in sorted(bleu_keys):
                print(f"  {key}: {results[key]:.4f}")
        
        # length statistics
        if "avg_pred_length" in results:
            print(f"\nlength statistics:")
            print(f"  average prediction length: {results['avg_pred_length']:.1f} words")
            if "avg_gt_length" in results:
                print(f"  average ground truth length: {results['avg_gt_length']:.1f} words")
    
    # detailed analysis
    if args.detailed_analysis and ground_truths and "error" not in results:
        print(f"\n" + "="*60)
        print("detailed analysis")
        print("="*60)
        
        # analyze prediction length distribution
        pred_lengths = [len(p.split()) for p in predictions]
        print(f"prediction length distribution:")
        print(f"  shortest: {min(pred_lengths)} words")
        print(f"  longest: {max(pred_lengths)} words")
        print(f"  median: {sorted(pred_lengths)[len(pred_lengths)//2]} words")
        
        # display some examples
        print(f"\nsample comparison (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"\nsample {i+1}:")
            print(f"  prediction: {predictions[i][:100]}{'...' if len(predictions[i]) > 100 else ''}")
            if i < len(ground_truths):
                print(f"  ground truth: {ground_truths[i][:100]}{'...' if len(ground_truths[i]) > 100 else ''}")
            print(f"  match: {'✓' if i < len(ground_truths) and evaluator.exact_match(predictions[i], ground_truths[i]) else '✗'}")
    
    # save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        result_file = output_dir / "response_evaluation_results.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"results saved to: {result_file}")
    
    # output JSON format results (for pipeline)
    print(json.dumps(results))

if __name__ == "__main__":
    main()