import os
import time
from openai import OpenAI
from typing import Union, List


def api_call(client,
             query: Union[List, str],
             model_name: str,
             max_tokens=1000,
             response_format='text',
             temperature=1.0):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": response_format}
            )
            resp = completion.choices[0].message.content
            return resp
        except Exception as e:
            print(messages)
            print(f"API_CALL Error: {e}")
            time.sleep(3)
            continue
    return ""


def get_client(model_name):
    if 'gpt' in model_name:
        API_KEY = os.getenv("GPT_API_KEY")
        BASE_URL_GPT = os.getenv("BASE_URL_GPT")
        gpt_client = OpenAI(base_url=BASE_URL_GPT, api_key=API_KEY)
        return gpt_client
    elif 'mistral' in model_name:
        # mistral's API are OpenAI-compatible
        API_KEY = os.getenv('MISTRAL_API_KEY')
        BASE_URL_MISTRAL = os.getenv("BASE_URL_MISTRAL",
                                     "https://api.mistral.ai/v1")
        mistral_client = OpenAI(base_url=BASE_URL_MISTRAL, api_key=API_KEY)
        return mistral_client
    elif 'claude' in model_name:
        # TODO: implement this client
        claude_client = ''
        return claude_client
    else:
        # TODO: implement this client
        llama_client = ''
        return llama_client
