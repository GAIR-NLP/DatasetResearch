#!/usr/bin/env python3
"""
convert the reading comprehension dataset to the instruction tuning format

input: a JSON file containing story, questions, and answers
output: a JSON file containing each question-answer pair as an independent sample
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_qa_pairs(data: List[Dict]) -> List[Dict]:
    """
    extract question-answer pairs from the original data
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: extracted question-answer pairs
    """
    qa_pairs = []
    
    for item_idx, item in enumerate(data):
        try:
            source = item.get('source', '')
            story = item.get('story', '')
            questions = item.get('questions', [])
            answers = item.get('answers', {})
            
            # get the answer text list
            answer_texts = answers.get('input_text', [])
            answer_starts = answers.get('answer_start', [])
            answer_ends = answers.get('answer_end', [])
            
            # ensure the number of questions and answers matches
            min_length = min(len(questions), len(answer_texts))
            
            if min_length == 0:
                logger.warning(f"⚠️ sample {item_idx} has no valid question-answer pairs")
                continue
            
            # create an independent sample for each question-answer pair
            for i in range(min_length):
                question = questions[i].strip()
                answer = answer_texts[i].strip()
                
                if question and answer:
                    qa_pair = {
                        "system": "Answer the question based on the given story context.",
                        "input": f"Story: {story}\n\nQuestion: {question}",
                        "output": answer,
                        "metadata": {
                            "source": source,
                            "story_id": item_idx,
                            "question_id": i,
                            "answer_start": answer_starts[i] if i < len(answer_starts) else None,
                            "answer_end": answer_ends[i] if i < len(answer_ends) else None,
                            "story_length": len(story),
                            "question_length": len(question)
                        }
                    }
                    
                    qa_pairs.append(qa_pair)
                    
                else:
                    logger.warning(f"⚠️ sample {item_idx} has an empty question-answer pair")
                    
        except Exception as e:
            logger.error(f"❌ error occurred when processing sample {item_idx}: {e}")
            continue
    
    return qa_pairs

def create_simple_qa_format(data: List[Dict]) -> List[Dict]:
    """
    create a simplified question-answer format
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: simplified question-answer pairs
    """
    simple_qa_pairs = []
    
    for item_idx, item in enumerate(data):
        try:
            story = item.get('story', '')
            questions = item.get('questions', [])
            answers = item.get('answers', {})
            
            # get the answer text list
            answer_texts = answers.get('input_text', [])
            
            # ensure the number of questions and answers matches
            min_length = min(len(questions), len(answer_texts))
            
            # create an independent sample for each question-answer pair
            for i in range(min_length):
                question = questions[i].strip()
                answer = answer_texts[i].strip()
                
                if question and answer:
                    simple_qa = {
                        "story": story,
                        "question": question,
                        "answer": answer,
                        "story_id": item_idx,
                        "question_id": i
                    }
                    
                    simple_qa_pairs.append(simple_qa)
                    
        except Exception as e:
            logger.error(f"❌ error occurred when processing sample {item_idx}: {e}")
            continue
    
    return simple_qa_pairs

def load_data(input_file: str) -> List[Dict]:
    """
    load JSON data
    
    Args:
        input_file: input file path
        
    Returns:
        List[Dict]: loaded data
    """
    logger.info(f"📖 reading file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                data = [data]
                
        logger.info(f"✅ successfully loaded {len(data)} data items")
        return data
        
    except Exception as e:
        logger.error(f"❌ file reading failed: {e}")
        return []

def convert_dataset(input_file: str, output_file: str, format_type: str = "instruction"):
    """
    convert dataset
    
    Args:
        input_file: input file path
        output_file: output file path
        format_type: output format type ("instruction" or "simple")
    """
    # read data
    data = load_data(input_file)
    
    if not data:
        logger.error("❌ no valid data read, exit conversion")
        return
    
    logger.info(f"📊 read {len(data)} original stories")
    
    # convert data
    if format_type == "instruction":
        converted_data = extract_qa_pairs(data)
        logger.info("📝 using instruction tuning format")
    else:
        converted_data = create_simple_qa_format(data)
        logger.info("📝 using simplified question-answer format")
    
    if not converted_data:
        logger.error("❌ no valid question-answer pairs generated")
        return
    
    # statistics
    total_stories = len(data)
    total_qa_pairs = len(converted_data)
    avg_qa_per_story = total_qa_pairs / total_stories if total_stories > 0 else 0
    
    # statistics of source distribution
    sources = {}
    for item in converted_data:
        if format_type == "instruction":
            source = item.get("metadata", {}).get("source", "unknown")
        else:
            # 对于简化格式，需要从原始数据获取source信息
            story_id = item.get("story_id", 0)
            source = data[story_id].get("source", "unknown") if story_id < len(data) else "unknown"
        sources[source] = sources.get(source, 0) + 1
    
    # save results
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ conversion completed!")
        logger.info(f"📊 statistics:")
        logger.info(f"   - number of original stories: {total_stories}")
        logger.info(f"   - number of generated question-answer pairs: {total_qa_pairs}")
        logger.info(f"   - average number of question-answer pairs per story: {avg_qa_per_story:.2f}")
        logger.info(f"   - data source distribution: {sources}")
        logger.info(f"💾 results saved to: {output_file}")
        
        # display example
        if converted_data:
            logger.info("\n📝 conversion result example:")
            example = converted_data[0]
            
            if format_type == "instruction":
                logger.info(f"System: {example.get('system', '')}")
                logger.info(f"Input: {example.get('input', '')[:300]}...")
                logger.info(f"Output: {example.get('output', '')}")
                logger.info(f"Metadata: {example.get('metadata', {})}")
            else:
                logger.info(f"Story: {example.get('story', '')[:200]}...")
                logger.info(f"Question: {example.get('question', '')}")
                logger.info(f"Answer: {example.get('answer', '')}")
                logger.info(f"Story ID: {example.get('story_id', '')}")
                logger.info(f"Question ID: {example.get('question_id', '')}")
            
    except Exception as e:
        logger.error(f"❌ saving results failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='extract question-answer pairs from the reading comprehension dataset')
    parser.add_argument('--input', '-i', required=True, help='input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='output file path')
    parser.add_argument('--format', '-f', choices=['instruction', 'simple'], 
                       default='instruction', help='输出格式 (instruction/simple)')
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.format)

if __name__ == "__main__":
    main() 