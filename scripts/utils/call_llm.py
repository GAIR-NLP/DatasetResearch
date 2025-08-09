import time
from functools import wraps
import openai

def retry_on_failure(max_retries=3, delay=1):
    """
    retry decorator
    
    Args:
        max_retries: maximum number of retries
        delay: retry interval (seconds)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries += 1
                    if retries < max_retries:
                        print(f"model {self.model} request failed (attempt {retries}/{max_retries}): {str(e)}")
                        print(f"{delay} seconds later retry...")
                        time.sleep(delay)
            
            print(f"model {self.model} failed after {max_retries} retries: {str(last_error)}")
            raise last_error
            
        return wrapper
    return decorator

class CallLLM:
    def __init__(self, model:str = "Qwen/Qwen2.5-7B-Instruct", 
                 api_base:str = "https://gpt.yunstorm.com/", 
                 api_key:str = "sk-your-azure-key"):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        
    @retry_on_failure(max_retries=10, delay=3)
    def post_request(self, messages:list) -> tuple[str, int]:
        """
        send request and get answer, with retry mechanism
        """
        if self.model in ['gpt-4o','gpt-4o-mini','o3','o3-mini','o1','o1-mini','o1-preview','o1-preview-mini','gpt-4.1','gpt-4.1-mini','gpt-4.1-nano']:
            client = openai.AzureOpenAI(azure_endpoint=self.api_base,api_key=self.api_key,api_version="2025-01-01-preview")
        else:
            client = openai.OpenAI(base_url=self.api_base,api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    
    # @retry_on_failure(max_retries=20, delay=4)
    # def post_search_request(self, messages:list) -> tuple[str, int]:
    #     """
    #     send request and get answer, with retry mechanism
    #     """
    #     client = openai.OpenAI(api_base=self.api_base,api_key=self.api_key)
    #     response = client.chat.completions.create(
    #         model=self.model,
    #         extra_body={},
    #         messages=messages,
    #     )
    #     return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    