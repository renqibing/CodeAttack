import time 
from utils import api_call, get_client

class TargetLLM:
    def __init__(self, model_name, max_tokens=0, seed=725, temperature=0.0):
        self.client = get_client(model_name)
        self.model_name = model_name
        self.max_retry = 3
        self.timeout = 200
        self.query_sleep = 20
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature

    def generate(self, query):
        for _ in range(1, self.max_retry + 1):
            try:
                resp = api_call(self.client, query, model_name=self.model_name)
                return resp
            except Exception as e:
                print("error", e)
                time.sleep(self.query_sleep)
        summ = f"All retry attempts failed."
        return summ  # Or raise an exception if desired