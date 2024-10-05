import re
import json
import pandas as pd


class DataPreparer:
    def __init__(self, query_file, prompt_name, prompt_type="python_stack"):
        self.query_file = query_file
        self.prompt_name = prompt_name
        self.prompt_type = prompt_type
        self._prompt = prompt_name
        self.query_name = query_file.split('/')[-1].split('.')[0]

    @property
    def prompt(self):
        """The prompt property, loaded from a file template."""
        return self._prompt

    @prompt.setter
    def prompt(self, prompt_name: str):
        filename = f"./prompt_templates/{prompt_name}"
        with open(filename, 'r') as file:
            prompt = file.read()
        return prompt

    def replace(self, filepath, replacement, replace_line=None):
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
        """This function processes the queries depending on the chosen prompt
        type, and then injects them into the template. The template depends on
        the prompt type, too."""
        if "python_stack" in self.prompt_name:
            # In a stack, elements are pushed in reverse order
            def format_and_push(input_string):
                # Split spaces and hyphens
                words = re.split(r'[\s\-]+', input_string)
                if len(words) == 1:
                    # handle the case where input is Chinese
                    # Split every character
                    words = list(words[0])
                    print(words)
                # Reverse order of words
                words = words[::-1]
                formatted_output = '\n'.join(f'    my_stack.append("{word}")'
                                             for word in words)
                return formatted_output
            wrapped_input = format_and_push(ori_query) + '\n'
        elif "python_list" in self.prompt_name:
            # In a list, elements are appended respecting their order
            def format_and_push(input_string):
                words = input_string.split()
                formatted_output = '\n'.join(f'    my_list.append("{word}")'
                                             for word in words)
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

        # Inject the wrapped input into the prompt template
        if "python" in self.prompt_name:
            # If the template is python, use python built-in format method to
            # inject the input replacing the placeholders
            prompt = self.prompt.format(wrapped_input=wrapped_input)
        else:
            # If the template is in a language other than python, use the
            # replace function from this class
            # TODO
            prompt = self.replace(f"./prompt_templates/{self.prompt_name}",
                                  wrapped_input,
                                  replacement_line)
        return prompt

    def infer(self):
        # TODO: If the prompt file already exists, avoid generating it again
        df = pd.read_csv(self.query_file)
        malicious_goals = df['goal'].tolist()
        # TODO: if it's not important to keep track of the index, then we can
        # avoid pre-generating this list of empty dict and we can just append
        # the results in the for loop
        results = [{} for _ in range(len(malicious_goals))]
        for idx, data in enumerate(malicious_goals):
            wrapped_res = {
                "plain_attack": data,
                f"code_wrapped_{self.prompt_type}": self.wrap(data)
            }
            results[idx] = wrapped_res

        results_dumped = json.dumps(results)
        with open(f'./prompts/data_{self.query_name}_{self.prompt_type}.json',
                  'w+') as f:
            f.write(results_dumped)
