import re
import json
import pandas as pd
from utils import read_prompt_from_file

class DataPreparer:
    def __init__(self, query_file, prompt_name, prompt_type="python_stack"):
        self.query_file = query_file
        self.prompt_name = prompt_name
        self.prompt_type = prompt_type
        self.prompt = read_prompt_from_file(f"./prompt_templates/{self.prompt_name}")
        self.query_name = query_file.split('/')[-1].split('.')[0]
        return 

    def replace(self, filepath=None, replacement=None, replace_line=None):
        prompt_prefix = ""
        prompt_prefix_file = open(filepath, 'r')
        for i, line in enumerate(prompt_prefix_file):
            if replace_line is not None and i == replace_line:
                prompt_prefix += replacement
            else:
                prompt_prefix += line
        if replace_line > i:
            prompt_prefix += replacement
        return prompt_prefix

    def wrap(self, ori_query):
        if "python_stack" in self.prompt_name:
            def format_and_push(input_string):
                words = re.split(r'[\s\-]+', input_string)
                if len(words) == 1:
                    # handle the case where input is Chinese
                    words = list(words[0])
                    print(words)
                words = words[::-1]
                formatted_output = '\n'.join(f'    my_stack.append("{word}")' for word in words)
                return formatted_output
            wrapped_input = format_and_push(ori_query) + '\n' 
        elif "python_list" in self.prompt_name:
            def format_and_push(input_string):
                words = input_string.split()
                formatted_output = '\n'.join(f'    my_list.append("{word}")' for word in words)
                return formatted_output
            wrapped_input = format_and_push(ori_query) + '\n'
        elif "python_string" in self.prompt_name:
            wrapped_input = f"    my_string = \"{ori_query}\"\n"
        elif "C_string" in self.prompt_name:
            wrapped_input = f"    std::string my_string = \"{ori_query}\";\n"
            replacement_line = 10
        elif "go_string" in self.prompt_name:
            wrapped_input = f"        myQueue := \"{ori_query}\"\n"
            replacement_line = 17

        if "python" in self.prompt_name:
            prompt = self.prompt.format(wrapped_input=wrapped_input)
        else:
            prompt = self.replace(f"./prompt_templates/{self.prompt_name}", wrapped_input, replacement_line)
        return prompt

    def infer(self, ):
        df = pd.read_csv(self.query_file)
        datas = df['goal'].tolist()
        results = [{} for _ in range(len(datas))]
        for idx, data in enumerate(datas):
            results[idx]["plain_attack"] = data
            results[idx][f"code_wrapped_{self.prompt_type}"] = self.wrap(data)

        results_dumped = json.dumps(results)
        with open(f'./prompts/data_{self.query_name}_{self.prompt_type}.json', 'w+') as f:
            f.write(results_dumped)
        f.close()
