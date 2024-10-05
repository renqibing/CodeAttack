import re


class PostProcessor:
    def __init__(self, prompt_type):
        self.prompt_type = prompt_type

    def core(self, response):
        if "python" in self.prompt_type:
            regex = r'output_list\.append\("([^"]+)"\)'
            parsed_res = re.findall(regex, response)
            if parsed_res:
                return parsed_res
            else:
                pattern = r"output_list\s*=\s*\[\s*(.*?)\s*\]"
                matches = re.search(pattern, response, re.DOTALL)
                if matches:
                    output_content = matches.group(1)
                    return output_content
                else:
                    return response
        elif "C" in self.prompt_type:
            regex = r'outputList\.push_back\("([^"]+)"\);'
            parsed_res = re.findall(regex, response)
            if parsed_res:
                return parsed_res
            else:
                return response
        elif "go" in self.prompt_type:
            regex = r'outputList\s*=\s*append\(outputList,\s*"([^"]+)"\)'
            parsed_res = re.findall(regex, response)
            if parsed_res:
                return parsed_res
            else:
                return response
        else:
            return response
