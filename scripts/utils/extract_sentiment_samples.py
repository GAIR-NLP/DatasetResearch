#!/usr/bin/env python3
"""
convert the sentiment analysis dataset to the instruction tuning format

input: a JSON file containing reviewId, sentences, and opinions
output: a JSON file containing each sentence with polarity as an independent sample
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_sentiment_samples(data: List[Dict]) -> List[Dict]:
    """
    extract sentences with polarity from the original data
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: extracted sentiment analysis samples
    """
    sentiment_samples = []
    
    for review_idx, review in enumerate(data):
        try:
            review_id = review.get('reviewId', f'review_{review_idx}')
            sentences = review.get('sentences', [])
            
            for sentence in sentences:
                sentence_id = sentence.get('sentenceId', '')
                text = sentence.get('text', '').strip()
                opinions = sentence.get('opinions', [])
                
                # only process sentences with opinions
                if opinions and text:
                    # create an independent sample for each opinion
                    for opinion_idx, opinion in enumerate(opinions):
                        category = opinion.get('category', {})
                        polarity = opinion.get('polarity', '')
                        
                        if polarity:  # ensure there is polarity
                            entity = category.get('entity', '')
                            attribute = category.get('attribute', '')
                            
                            sample = {
                                "system": "Analyze the sentiment of the given text towards the specified entity and attribute.",
                                "input": f"Text: {text}\nEntity: {entity}\nAttribute: {attribute}",
                                "output": polarity,
                                "metadata": {
                                    "review_id": review_id,
                                    "sentence_id": sentence_id,
                                    "opinion_index": opinion_idx,
                                    "entity": entity,
                                    "attribute": attribute,
                                    "category": f"{entity}#{attribute}",
                                    "text_length": len(text)
                                }
                            }
                            
                            sentiment_samples.append(sample)
                            
        except Exception as e:
            logger.error(f"❌ error occurred when processing review {review_idx}: {e}")
            continue
    
    return sentiment_samples

def create_simple_sentiment_format(data: List[Dict]) -> List[Dict]:
    """
    create a simplified sentiment analysis format
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: simplified sentiment analysis samples
    """
    simple_samples = []
    
    for review_idx, review in enumerate(data):
        try:
            review_id = review.get('reviewId', f'review_{review_idx}')
            sentences = review.get('sentences', [])
            
            for sentence in sentences:
                sentence_id = sentence.get('sentenceId', '')
                text = sentence.get('text', '').strip()
                opinions = sentence.get('opinions', [])
                
                # only process sentences with opinions
                if opinions and text:
                    # extract all polarities and categories
                    polarities = []
                    categories = []
                    entities = []
                    attributes = []
                    
                    for opinion in opinions:
                        polarity = opinion.get('polarity', '')
                        if polarity:
                            category = opinion.get('category', {})
                            entity = category.get('entity', '')
                            attribute = category.get('attribute', '')
                            
                            polarities.append(polarity)
                            categories.append(f"{entity}#{attribute}")
                            entities.append(entity)
                            attributes.append(attribute)
                    
                    if polarities:  # ensure there is valid polarity
                        simple_sample = {
                            "text": text,
                            "polarities": polarities,
                            "categories": categories,
                            "entities": entities,
                            "attributes": attributes,
                            "review_id": review_id,
                            "sentence_id": sentence_id,
                            "num_opinions": len(polarities)
                        }
                        
                        simple_samples.append(simple_sample)
                        
        except Exception as e:
            logger.error(f"❌ error occurred when processing review {review_idx}: {e}")
            continue
    
    return simple_samples

def create_single_opinion_format(data: List[Dict]) -> List[Dict]:
    """
    create a single opinion format (each opinion as an independent sample)
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: single opinion samples
    """
    single_opinion_samples = []
    
    for review_idx, review in enumerate(data):
        try:
            review_id = review.get('reviewId', f'review_{review_idx}')
            sentences = review.get('sentences', [])
            
            for sentence in sentences:
                sentence_id = sentence.get('sentenceId', '')
                text = sentence.get('text', '').strip()
                opinions = sentence.get('opinions', [])
                
                # only process sentences with opinions
                if opinions and text:
                    for opinion_idx, opinion in enumerate(opinions):
                        polarity = opinion.get('polarity', '')
                        
                        if polarity:  # ensure there is polarity
                            category = opinion.get('category', {})
                            entity = category.get('entity', '')
                            attribute = category.get('attribute', '')
                            
                            single_sample = {
                                "text": text,
                                "entity": entity,
                                "attribute": attribute,
                                "category": f"{entity}#{attribute}",
                                "polarity": polarity,
                                "review_id": review_id,
                                "sentence_id": sentence_id,
                                "opinion_index": opinion_idx
                            }
                            
                            single_opinion_samples.append(single_sample)
                            
        except Exception as e:
            logger.error(f"❌ error occurred when processing review {review_idx}: {e}")
            continue
    
    return single_opinion_samples

def load_data(input_file: str) -> List[Dict]:
    """
    load JSON data, support single line JSON and JSONL format
    
    Args:
        input_file: input file path
        
    Returns:
        List[Dict]: loaded data
    """
    logger.info(f"📖 reading file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # first try to load as standard JSON
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                logger.info(f"✅ successfully loaded data as JSON")
            except json.JSONDecodeError:
                # if failed, try JSONL format
                f.seek(0)
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ failed to parse JSON at line {line_num}: {e}")
                logger.info(f"✅ successfully loaded data as JSONL")
                
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
        format_type: output format type ("instruction", "simple", "single")
    """
    # read data
    data = load_data(input_file)
    
    if not data:
        logger.error("❌ no valid data read, exit conversion")
        return
    
    logger.info(f"📊 read {len(data)} original reviews")
    
    # convert data
    if format_type == "instruction":
        converted_data = extract_sentiment_samples(data)
        logger.info("📝 using instruction tuning format")
    elif format_type == "simple":
        converted_data = create_simple_sentiment_format(data)
        logger.info("📝 using simplified sentiment analysis format")
    else:  # single
        converted_data = create_single_opinion_format(data)
        logger.info("📝 using single opinion format")
    
    if not converted_data:
        logger.error("❌ no valid sentiment analysis samples generated")
        return
    
    # statistics
    total_reviews = len(data)
    total_samples = len(converted_data)
    
    # statistics of polarity distribution
    polarity_counts = {}
    for item in converted_data:
        if format_type == "instruction":
            polarity = item.get("output", "unknown")
        elif format_type == "simple":
            polarities = item.get("polarities", [])
            for p in polarities:
                polarity_counts[p] = polarity_counts.get(p, 0) + 1
            continue
        else:  # single
            polarity = item.get("polarity", "unknown")
        polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
    
    # 保存结果
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ conversion completed!")
        logger.info(f"📊 statistics:")
        logger.info(f"   - number of original reviews: {total_reviews}")
        logger.info(f"   - number of generated samples: {total_samples}")
        logger.info(f"   - sentiment polarity distribution: {polarity_counts}")
        logger.info(f"💾 results saved to: {output_file}")
        
        # display example
        if converted_data:
            logger.info("\n📝 conversion result example:")
            example = converted_data[0]
            
            if format_type == "instruction":
                logger.info(f"System: {example.get('system', '')}")
                logger.info(f"Input: {example.get('input', '')[:200]}...")
                logger.info(f"Output: {example.get('output', '')}")
                logger.info(f"Metadata: {example.get('metadata', {})}")
            elif format_type == "simple":
                logger.info(f"Text: {example.get('text', '')[:200]}...")
                logger.info(f"Polarities: {example.get('polarities', [])}")
                logger.info(f"Categories: {example.get('categories', [])}")
            else:  # single
                logger.info(f"Text: {example.get('text', '')[:200]}...")
                logger.info(f"Entity: {example.get('entity', '')}")
                logger.info(f"Attribute: {example.get('attribute', '')}")
                logger.info(f"Polarity: {example.get('polarity', '')}")
            
    except Exception as e:
        logger.error(f"❌ saving results failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='extract sentiment analysis samples from the dataset')
    parser.add_argument('--input', '-i', required=True, help='input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='output file path')
    parser.add_argument('--format', '-f', choices=['instruction', 'simple', 'single'], 
                       default='single', help='output format (instruction/simple/single)')
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.format)

if __name__ == "__main__":
    main() 