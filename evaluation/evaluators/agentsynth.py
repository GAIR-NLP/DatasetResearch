#!/usr/bin/env python3
"""
    integrated Web task evaluation system
    integrate external dependencies from original code into a standalone script
"""

import json
import random
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# set log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== config classes ====================

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    api_key: str
    base_url: str
    max_tokens: int = 4000
    temperature: float = 0.1

@dataclass
class BrowserConfig:
    """browser config"""
    headless: bool = True
    timeout: int = 30
    window_size: Tuple[int, int] = (1920, 1080)
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

@dataclass
class EvaluationConfig:
    """evaluation config"""
    max_actions: int = 10
    max_workers: int = 5
    retry_attempts: int = 3
    success_threshold: float = 0.9

# ==================== abstract base classes ====================

class AIClient(ABC):
    """AI client abstract base class"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """generate response"""
        pass

class WebEnvironment(ABC):
    """Web environment abstract base class"""
    
    @abstractmethod
    def navigate(self, url: str) -> bool:
        """navigate to URL"""
        pass
    
    @abstractmethod
    def take_screenshot(self) -> bytes:
        """take screenshot"""
        pass
    
    @abstractmethod
    def execute_action(self, action: str) -> bool:
        """execute action"""
        pass
    
    @abstractmethod
    def close(self):
        """close environment"""
        pass

# ==================== concrete implementation classes ====================

class MockAIClient(AIClient):
    """mock AI client (for testing)"""
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """mock generate response"""
        time.sleep(0.1)  # mock API call delay
        return f"Mock response for model {self.config.model_name}"

class RealAIClient(AIClient):
    """real AI client"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = self._create_client()
    
    def _create_client(self):
        """create specific AI client"""
        try:
            if self.config.model_name.startswith("gpt") or self.config.model_name.startswith("o4"):
                import openai
                return openai.OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url if self.config.base_url else None
                )
            # elif self.config.model_name.startswith("claude"):
            #     import anthropic
            #     return anthropic.Anthropic(
            #         api_key=self.config.api_key,
            #         base_url=self.config.base_url
            #     )
            # elif self.config.model_name.startswith("gemini"):
            #     import google.generativeai as genai
            #     genai.configure(api_key=self.config.api_key)
            #     return genai
            else:
                raise ValueError(f"unsupported model: {self.config.model_name}")
        except ImportError as e:
            logger.warning(f"failed to import AI client library: {e}, using mock client")
            return None
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """generate response"""
        if self.client is None:
            return f"mock response for {self.config.model_name}"
        
        try:
            if self.config.model_name.startswith("gpt") or self.config.model_name.startswith("o4"):
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                return response.choices[0].message.content
            elif self.config.model_name.startswith("claude"):
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.config.model_name.startswith("gemini"):
                model = self.client.GenerativeModel(self.config.model_name)
                response = model.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.error(f"AI client call failed: {e}")
            return f"error response: {str(e)}"

class MockWebEnvironment(WebEnvironment):
    """mock web environment"""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self.current_url = ""
        self.is_closed = False
    
    def navigate(self, url: str) -> bool:
        """mock navigate"""
        if self.is_closed:
            return False
        time.sleep(0.1)  # mock navigate delay
        self.current_url = url
        return True
    
    def take_screenshot(self) -> bytes:
        """mock take screenshot"""
        return b"fake_screenshot_data"
    
    def execute_action(self, action: str) -> bool:
        """mock execute action"""
        time.sleep(0.1)
        return random.random() > 0.3  # 70% success rate
    
    def close(self):
        """close environment"""
        self.is_closed = True

class RealWebEnvironment(WebEnvironment):
    """real web environment (using Selenium)"""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self.driver = self._create_driver()
    
    def _create_driver(self):
        """create browser driver"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            options = Options()
            if self.config.headless:
                options.add_argument("--headless")
            options.add_argument(f"--window-size={self.config.window_size[0]},{self.config.window_size[1]}")
            options.add_argument(f"--user-agent={self.config.user_agent}")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=options)
            driver.implicitly_wait(self.config.timeout)
            return driver
        except ImportError as e:
            logger.warning(f"failed to import Selenium: {e}, using mock environment")
            return None
    
    def navigate(self, url: str) -> bool:
        """navigate to URL"""
        if self.driver is None:
            return False
        try:
            self.driver.get(url)
            return True
        except Exception as e:
            logger.error(f"navigate failed: {e}")
            return False
    
    def take_screenshot(self) -> bytes:
        """take screenshot"""
        if self.driver is None:
            return b"fake_screenshot"
        try:
            return self.driver.get_screenshot_as_png()
        except Exception as e:
            logger.error(f"take screenshot failed: {e}")
            return b"error_screenshot"
    
    def execute_action(self, action: str) -> bool:
        """execute action"""
        if self.driver is None:
            return False
        try:
            # here we need to parse and execute the action based on the specific action format
            # simplified implementation, actually we need more complex action parsing logic
            return True
        except Exception as e:
            logger.error(f"action execution failed: {e}")
            return False
    
    def close(self):
        """close environment"""
        if self.driver:
            self.driver.quit()

# ==================== core evaluation system ====================

class TrajectoryGenerator:
    """trajectory generator"""
    
    def __init__(self, agent_client: AIClient, judge_client: AIClient):
        self.agent_client = agent_client
        self.judge_client = judge_client
    
    def generate_trajectory(self, 
                          env: WebEnvironment, 
                          url: str, 
                          instruction: str, 
                          max_actions: int = 10) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """generate task execution trajectory"""
        observations = []
        actions = []
        
        # navigate to target URL
        if not env.navigate(url):
            return observations, actions, {"success": 0.0, "error": "navigate failed"}
        
        # initial observation
        screenshot = env.take_screenshot()
        observations.append(f"initial page screenshot: {len(screenshot)} bytes")
        
        # execute action sequence
        for i in range(max_actions):
            # generate next action
            prompt = f"""
            任务: {instruction}
            当前步骤: {i+1}/{max_actions}
            页面状态: {observations[-1]}
            
            请生成下一步动作:
            """
            
            action_response = self.agent_client.generate_response(prompt)
            actions.append(action_response)
            
            # execute action
            success = env.execute_action(action_response)
            if not success:
                observations.append(f"action execution failed: {action_response}")
                break
            
            # get new observation
            new_screenshot = env.take_screenshot()
            observations.append(f"after action screenshot: {len(new_screenshot)} bytes")
            
            # check if task is complete
            if self._is_task_complete(instruction, observations, actions):
                break
        
        # judge if task is successful
        judgment = self._judge_task_success(instruction, observations, actions)
        
        return observations, actions, judgment
    
    def _is_task_complete(self, instruction: str, observations: List[str], actions: List[str]) -> bool:
        """judge if task is complete"""
        # simplified implementation, actually we need more complex judgment logic
        return len(actions) >= 3 and random.random() > 0.7
    
    def _judge_task_success(self, instruction: str, observations: List[str], actions: List[str]) -> Dict[str, Any]:
        """judge if task is successful"""
        prompt = f"""
        task: {instruction}
        observation sequence: {observations}
        action sequence: {actions}
        
        please judge if task is successful, return success rate (0-1) and reason:
        """
        
        judgment_response = self.judge_client.generate_response(prompt)
        
        # simplified judgment logic
        success_score = random.uniform(0.0, 1.0)
        
        return {
            "success": success_score,
            "task_success": success_score >= 0.9,
            "judgment": judgment_response,
            "total_actions": len(actions),
            "total_observations": len(observations)
        }

class TaskEvaluator:
    """task evaluator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results_cache = {}
    
    def create_ai_client(self, model_name: str, secrets: Dict[str, str]) -> AIClient:
        """create AI client"""
        if model_name.startswith("claude"):
            model_config = ModelConfig(
                model_name=model_name,
                api_key=secrets.get("ANTHROPIC_API_KEY", ""),
                base_url="https://api.anthropic.com/v1"
            )
        elif model_name.startswith("gemini"):
            model_config = ModelConfig(
                model_name=model_name,
                api_key=secrets.get("GOOGLE_API_KEY", ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai"
            )
        else:  # OpenAI models
            model_config = ModelConfig(
                model_name=model_name,
                api_key=secrets.get("OPENAI_API_KEY", ""),
                base_url=""
            )
        
        return RealAIClient(model_config)
    
    def create_web_environment(self) -> WebEnvironment:
        """create web environment"""
        browser_config = BrowserConfig()
        return MockWebEnvironment(browser_config)  # default using mock environment
    
    def evaluate_single_task(self, task_data: Dict[str, Any], model_name: str, secrets: Dict[str, str]) -> bool:
        """evaluate single task"""
        task_text = task_data.get("task") or task_data.get("description", "")
        website = task_data.get("website", "")
        
        logger.info(f"evaluate task - website: {website}, model: {model_name}")
        
        try:
            # create client and environment
            agent_client = self.create_ai_client(model_name, secrets)
            judge_client = self.create_ai_client(model_name, secrets)
            env = self.create_web_environment()
            
            # generate trajectory
            trajectory_generator = TrajectoryGenerator(agent_client, judge_client)
            observations, actions, judgment = trajectory_generator.generate_trajectory(
                env=env,
                url=website,
                instruction=task_text,
                max_actions=self.config.max_actions
            )
            
            # judge if task is successful
            success = False
            if judgment and isinstance(judgment, dict):
                success_value = judgment.get('success', 0.0)
                task_success = judgment.get('task_success', False)
                success = (success_value >= self.config.success_threshold) or task_success
            
            logger.info(f"task result: {'success' if success else 'failed'}")
            
            # clean up
            env.close()
            
            return success
            
        except Exception as e:
            logger.error(f"task evaluation error: {e}")
            return False
    
    def evaluate_task_sequence(self, tasks: List[Dict[str, Any]], model_name: str, secrets: Dict[str, str]) -> List[bool]:
        """evaluate task sequence"""
        results = []
        
        # create shared environment
        env = self.create_web_environment()
        
        try:
            for idx, task in enumerate(tasks):
                if idx > 0 and not results[-1]:
                    # previous task failed, skip subsequent tasks
                    logger.info(f"skip task {idx+1}, because previous task failed")
                    results.append(False)
                    continue
                
                task_text = task.get("task") or task.get("description", "")
                website = task.get("website", "")
                
                try:
                    # create client
                    agent_client = self.create_ai_client(model_name, secrets)
                    judge_client = self.create_ai_client(model_name, secrets)
                    
                    # generate trajectory
                    trajectory_generator = TrajectoryGenerator(agent_client, judge_client)
                    observations, actions, judgment = trajectory_generator.generate_trajectory(
                        env=env,
                        url=website,
                        instruction=task_text,
                        max_actions=self.config.max_actions
                    )
                    
                    # judge if task is successful
                    success = False
                    if judgment and isinstance(judgment, dict):
                        success_value = judgment.get('success', 0.0)
                        task_success = judgment.get('task_success', False)
                        success = (success_value >= self.config.success_threshold) or task_success
                    
                    results.append(success)
                    
                except Exception as e:
                    logger.error(f"sequence task {idx+1} execution error: {e}")
                    results.append(False)
        
        finally:
            env.close()
        
        return results

# ==================== utility functions ====================

def load_jsonl(filename: str) -> List[Dict[str, Any]]:
    """load JSONL file"""
    tasks = []
    try:
        with open(filename, "r", encoding="utf8") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    except FileNotFoundError:
        logger.warning(f"file not found: {filename}")
    return tasks

def save_result_to_jsonl(result_entry: Dict[str, Any], filename: str):
    """save result to JSONL file"""
    with open(filename, "a", encoding="utf8") as outfile:
        json.dump(result_entry, outfile, ensure_ascii=False)
        outfile.write("\n")

def save_final_results(results: Dict[str, Any], filename: str):
    """save final results"""
    with open(filename, "w", encoding="utf8") as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)
    logger.info(f"final results saved to: {filename}")

def group_tasks_by_sequence(tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """group tasks by sequence"""
    if not tasks:
        return []
    
    sequences = []
    current_sequence = [tasks[0]]
    current_website = tasks[0].get("website")
    
    for task in tasks[1:]:
        website = task.get("website")
        if website == current_website:
            current_sequence.append(task)
        else:
            sequences.append(current_sequence)
            current_sequence = [task]
            current_website = website
    
    sequences.append(current_sequence)
    return sequences

def load_existing_results(filename: str) -> Optional[Dict[str, List[Any]]]:
    """load existing results"""
    try:
        results = {}
        with open(filename, "r", encoding="utf8") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    model = entry.get("model")
                    task_idx = entry.get("task_idx")
                    success = entry.get("success")
                    
                    if model not in results:
                        results[model] = []
                    
                    while len(results[model]) <= task_idx:
                        results[model].append(None)
                    
                    results[model][task_idx] = success
        return results
    except FileNotFoundError:
        return None

# ==================== main function ====================

def main():
    """main function"""
    # config
    models_to_evaluate = ["gpt-4o", "claude-3-sonnet-20240229"]  # using actual available models
    evaluation_config = EvaluationConfig(max_workers=3)  # reduce concurrency
    
    # create output directory
    os.makedirs("results", exist_ok=True)
    
    # load secrets
    try:
        with open("secrets.json", "r") as f:
            secrets = json.load(f)
    except FileNotFoundError:
        logger.warning("secrets.json file not found, using empty config")
        secrets = {}
    
    # create evaluator
    evaluator = TaskEvaluator(evaluation_config)
    
    # mock data (if no real data file)
    if not os.path.exists("summarized_task_sequences.jsonl"):
        logger.info("create mock task data")
        sample_tasks = [
            {"task": "搜索Python教程", "website": "https://www.google.com"},
            {"task": "查看首页", "website": "https://www.python.org"},
            {"task": "登录账户", "website": "https://github.com"},
        ]
        with open("summarized_task_sequences.jsonl", "w", encoding="utf8") as f:
            for task in sample_tasks:
                json.dump(task, f, ensure_ascii=False)
                f.write("\n")
    
    # ========== evaluate summary tasks ==========
    logger.info("start evaluating summary tasks")
    summarized_tasks = load_jsonl("summarized_task_sequences.jsonl")
    
    if summarized_tasks:
        summarized_results = {}
        
        for model in models_to_evaluate:
            logger.info(f"using model {model} to evaluate summary tasks")
            summarized_results[model] = []
            
            with ThreadPoolExecutor(max_workers=evaluation_config.max_workers) as executor:
                futures = {
                    executor.submit(evaluator.evaluate_single_task, task, model, secrets): idx
                    for idx, task in enumerate(summarized_tasks)
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"总结任务-{model}"):
                    idx = futures[future]
                    try:
                        success = future.result()
                        summarized_results[model].append(success)
                        
                        # save intermediate results
                        save_result_to_jsonl({
                            "model": model,
                            "task_idx": idx,
                            "success": success
                        }, "results/summarized_results.jsonl")
                        
                    except Exception as e:
                        logger.error(f"task {idx} execution failed: {e}")
                        summarized_results[model].append(False)
        
        # save final results
        save_final_results(summarized_results, "results/summarized_results_final.json")
        
        # print results
        logger.info("=== summary tasks evaluation results ===")
        for model, results in summarized_results.items():
            success_count = sum(results)
            total = len(results)
            logger.info(f"{model}: {success_count}/{total} ({success_count/total*100:.1f}%)")
    
    logger.info("evaluation completed!")

if __name__ == "__main__":
    main()