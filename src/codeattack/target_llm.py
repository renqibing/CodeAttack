import time
from utils import api_call, get_client


class TargetLLM:
    def __init__(self, model_name, max_tokens=512, seed=725, temperature=0.0):
        self.client = get_client(model_name)
        self.model_name = model_name
        self.max_retry = 3
        self.timeout = 200
        self.query_sleep = 20
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature

    def generate(self, query):
        for _ in range(self.max_retry):
            try:
                resp = api_call(client=self.client,
                                query=query,
                                model_name=self.model_name,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens)
                return resp
            except Exception as e:
                print("error", e)
                time.sleep(self.query_sleep)
        summ = "All retry attempts failed."
        return summ  # Or raise an exception if desired
