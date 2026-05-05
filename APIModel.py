
import os
from openai import OpenAI


class ApiModel():
    def __init__(self, model_type, api_key=None, base_url=None, use_proxy=False,init_params=None, **kwargs):

        self.model_type = model_type.replace('<api>', '')

        if use_proxy:
            os.environ["ALL_PROXY"] = "socks5://127.0.0.1:7891"

        self.client = OpenAI(base_url=base_url,api_key=api_key)

        self.init_params = {
            "temperature": 0.,
            "top_p": 0.9,
            "max_tokens": 10000,
            "logprobs": False,
            "presence_penalty": 0,
            "frequency_penalty": 0.1,
        }
        if init_params is not None:
            self.init_params.update(init_params)

    def answer(self, messages, run_params=None, response_format=None, verbose=False, **kwargs):

        used_params = self.init_params.copy()
        if run_params is not None:
            used_params.update(run_params)

        # 调用API以进行对话
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_type,
            temperature=used_params['temperature'],
            top_p=used_params['top_p'],
            presence_penalty=used_params['presence_penalty'],
            frequency_penalty=used_params['frequency_penalty'],
            response_format=response_format,
            max_completion_tokens=used_params['max_tokens'],
            logprobs=used_params.get('logprobs'),
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        response_content = response.choices[0].message.content

        if verbose:
            print(response)

        if used_params.get('logprobs'):
            try:
                response_logprobs = response.choices[0].message.logprobs
            except AttributeError:
                response_logprobs = response.choices[0].logprobs.content

            return response_content, response_logprobs
        else:
            return response_content, None
