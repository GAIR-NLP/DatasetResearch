#!/usr/bin/env python3
"""
crawl abstracts from URLs in CSV file

input: CSV file with URL column
output: CSV file with abstract column
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import argparse
import logging
from pathlib import Path
import time
from typing import Optional, Dict
import json
from urllib.parse import urlparse
import random

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# set request headers
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
]

def get_random_headers() -> Dict[str, str]:
    """
    generate random request headers
    """
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def extract_abstract(html_content: str) -> Optional[str]:
    """
    extract abstract from HTML content
    
    Args:
        html_content: HTML content of the webpage
        
    Returns:
        Optional[str]: extracted abstract text, if not found then return None
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # find abstract block
        abstract_block = soup.find('blockquote', class_='abstract')
        if not abstract_block:
            return None
            
        # remove "Abstract:" label
        descriptor = abstract_block.find('span', class_='descriptor')
        if descriptor:
            descriptor.decompose()
            
        # get cleaned text
        abstract_text = abstract_block.get_text(strip=True)
        return abstract_text
        
    except Exception as e:
        logger.error(f"error parsing HTML: {e}")
        return None

def fetch_url(url: str, max_retries: int = 3, delay: float = 1.0) -> Optional[str]:
    """
    get URL content, support retry
    
    Args:
        url: URL to crawl
        max_retries: maximum number of retries
        delay: request interval time (seconds)
        
    Returns:
        Optional[str]: webpage content, if failed then return None
    """
    for attempt in range(max_retries):
        try:
            # add random delay to avoid request too fast
            time.sleep(delay * (1 + random.random()))
            
            response = requests.get(url, headers=get_random_headers(), timeout=10)
            response.raise_for_status()
            return response.text
            
        except requests.RequestException as e:
            logger.warning(f"{attempt + 1}th request failed: {url} - {e}")
            if attempt == max_retries - 1:
                logger.error(f"reach maximum number of retries, give up request: {url}")
                return None
            continue

def process_urls(input_file: str, output_file: str, url_column: str = 'url'):
    """
    process URLs in CSV file and crawl abstracts
    
    Args:
        input_file: input CSV file path
        output_file: output CSV file path
        url_column: name of the URL column
    """
    try:
        # read CSV file
        logger.info(f"📖 reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        if url_column not in df.columns:
            logger.error(f"❌ URL column not found: {url_column}")
            return
            
        total_urls = len(df)
        logger.info(f"📊 found {total_urls} URLs")
        
        # add new column for storing abstract
        df['abstract'] = None
        df['crawl_status'] = None
        
        # create progress log file
        progress_file = Path(output_file).with_suffix('.progress.json')
        
        # process each URL
        success_count = 0
        fail_count = 0
        
        for idx, row in df.iterrows():
            try:
                url = row[url_column].replace( "https://arxiv.org/pdf/","https://arxiv.org/abs/")
            except:
                df.at[idx, 'crawl_status'] = 'invalid_url'
                fail_count += 1
                logger.warning(f"⚠️ skip invalid URL: {url}")
                continue
            if not isinstance(url, str) or not url.strip():
                logger.warning(f"⚠️ skip invalid URL: {url}")
                df.at[idx, 'crawl_status'] = 'invalid_url'
                continue
                
            logger.info(f"🌐 processing [{idx + 1}/{total_urls}]: {url}")
            
            # get webpage content
            html_content = fetch_url(url)
            if not html_content:
                logger.error(f"❌ get page failed: {url}")
                df.at[idx, 'crawl_status'] = 'fetch_failed'
                fail_count += 1
                continue
                
            # extract abstract
            abstract = extract_abstract(html_content)
            if abstract:
                df.at[idx, 'abstract'] = abstract
                df.at[idx, 'crawl_status'] = 'success'
                success_count += 1
                logger.info(f"✅ success extract abstract: {abstract[:100]}...")
            else:
                df.at[idx, 'crawl_status'] = 'extract_failed'
                fail_count += 1
                logger.warning(f"⚠️ no abstract found: {url}")
            
            # save progress periodically
            if (idx + 1) % 10 == 0:
                progress = {
                    'total': total_urls,
                    'processed': idx + 1,
                    'success': success_count,
                    'failed': fail_count,
                    'last_url': url,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress, f, ensure_ascii=False, indent=2)
                    
                # save current results
                df.to_csv(output_file, index=False)
                logger.info(f"💾 save progress: {idx + 1}/{total_urls}")
        
        # save final results
        df.to_csv(output_file, index=False)
        
        # output statistics
        logger.info("\n📊 crawl statistics:")
        logger.info(f"   - total URLs: {total_urls}")
        logger.info(f"   - success count: {success_count}")
        logger.info(f"   - failed count: {fail_count}")
        logger.info(f"   - success rate: {success_count/total_urls*100:.2f}%")
        
        # show status distribution
        status_counts = df['crawl_status'].value_counts()
        logger.info("\n📊 status distribution:")
        for status, count in status_counts.items():
            logger.info(f"   - {status}: {count}")
        
        logger.info(f"💾 results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ error during processing: {e}")

def main():
    parser = argparse.ArgumentParser(description='crawl abstracts from URLs')
    parser.add_argument('--input', '-i', required=True, help='input CSV file path')
    parser.add_argument('--output', '-o', required=True, help='output CSV file path')
    parser.add_argument('--url-column', default='bench_paper_url', help='name of the URL column (default: url)')
    
    args = parser.parse_args()
    process_urls(args.input, args.output, args.url_column)

if __name__ == "__main__":
    main() 