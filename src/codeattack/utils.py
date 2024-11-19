import os
import time
from openai import OpenAI
from typing import Union, List


# Implement your gpt client here
API_KEY = os.getenv("GPT_API_KEY")
BASE_URL_GPT = os.getenv("BASE_URL_GPT")
gpt_client = OpenAI(base_url=BASE_URL_GPT, api_key=API_KEY)
# Implement your claude client here
claude_client = ""
# Implement your llama client here
llama_client = ""


def api_call(client, query: Union[List, str],
             model_name="gpt-4o",
             response_format='text',
             temperature=1):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    for _ in range(3):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1000,
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
        return gpt_client
    elif 'claude' in model_name:
        return claude_client
    else:
        return llama_client


if __name__ == '__main__':
    query = "I have an array a with length of n, the function f(l, r) represents the sum of numbers in a from index l to r. I can re-sort the array a, to minimize the sum of all intervals"
    # model_name = "claude-3-5-sonnet-20240620"
    model_name = "gpt-4o"
    client = get_client(model_name)
    print(api_call(client, query, model_name))
