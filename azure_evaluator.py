# rewritten by Ruilin Cai
# based on YiVal original code, adding supports of azure models

import copy
import logging
import os
import string
from typing import Any, Dict, Iterable, List, Optional, Union

import aiohttp
# for exponential backoff
import openai
from openai import AzureOpenAI
from aiohttp_socks import ProxyConnector  # type: ignore
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

from yival.evaluators.base_evaluator import BaseEvaluator
from yival.schemas.evaluator_config import (
    EvaluatorOutput,
)
from yival.schemas.experiment_config import ExperimentResult

# custom support
from azure_evaluator_config import AzureEvaluatorConfig
from AzureSelfLibrary import create_assistant, create_assistant_and_get_response, client_response

CLASSIFY_STR = """
First, write out in a step by step manner your reasoning to be sure that your
conclusion is correct.
Avoid simply stating the correct answer at the outset.
Then print only a single choice from {choices} (without quotes or punctuation)
on its own line corresponding to the correct answer.
At the end, repeat just the answer by itself on a new line.
Reasoning:
"""

MATCH_FNS = {
    "include": lambda x, y: float(x in y),
    "exact": lambda x, y: float(x == y),
    "endswith": lambda x, y: x.endswith(y),
    "starts_or_endswith": lambda x, y: x.startswith(y) or x.endswith(y),
}

def extract_choice_from_response(
    response: str, choice_strings: Iterable[str]
) -> str:
    """Extracts the choice from the response string."""
    try:
        lines = response.strip().split("\n")
        for line in lines:
            sanitized_line = "".join(
                c for c in line if c not in string.punctuation
            ).strip()
            if not sanitized_line:
                continue
            for choice in choice_strings:
                if MATCH_FNS["exact"](sanitized_line, choice):
                    return choice
    except:
        return "invalid response"


def calculate_choice_score(
    choice: str,
    choice_scores: Optional[Dict[str, float]] = None
) -> Optional[float]:
    """Calculates the score for the given choice."""
    if choice_scores is None:
        return None
    if choice == "invalid response" or choice is None:
        return min(choice_scores.values())
    return choice_scores.get(choice)


def format_template(
    template: Union[str, List[Dict[str, str]]], content: Dict[str, Any]
) -> Union[str, List[Dict[str, str]]]:
    """Formats a string or list template with the provided content."""
    if isinstance(template, str):
        try:
            return template.format(**content)
        except KeyError as e:
            raise ValueError(f"Missing key {e} in content dictionary")

    res = []
    for t in template:
        formatted_msg = copy.deepcopy(t)
        try:
            if "content" in formatted_msg:
                formatted_msg["content"] = formatted_msg['content'].format(
                    **content
                )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in content dictionary")
        res.append(formatted_msg)
    return res


@retry(
    wait=wait_random(min=1, max=20),
    stop=stop_after_attempt(100),
    before_sleep=before_sleep_log(logger, logging.DEBUG)
)
def completion_with_backpff(**kwargs):
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2023-12-01-preview"
    )
    response = client.chat.completions.create(**kwargs)
    return response


@retry(
    wait=wait_random(min=1, max=20),
    stop=stop_after_attempt(100),
)
async def acompletion_with_backpff(**kwargs):
    print("enter async!!!\n")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json"
    }

    proxy = os.environ.get("all_proxy")
    if proxy:
        connector = ProxyConnector.from_url(proxy)
    else:
        connector = None
    kwargs.pop('request_timeout', None)

    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.post(url, headers=headers, json=kwargs) as response:

            return await response.json()


def choices_to_string(choice_strings: Iterable[str]) -> str:
    """Converts a list of choices into a formatted string."""
    return " or ".join(f'"{choice}"' for choice in choice_strings)


class AzureEvaluator(BaseEvaluator):
    """Evaluator using OpenAI's prompt-based evaluation."""

    default_config = AzureEvaluatorConfig(
        name="azure_evaluator"
    )

    def __init__(self, config: AzureEvaluatorConfig):
        super().__init__(config)
        self.config = config
        # for azure openai assistant, currently not working
        # self.assistant_id = create_assistant(
        #     name="Evaluator",
        #     instructions="You are a helpful assistant that evaluate given text content following specific criteria.",
        #     tools=[],
        #     model=os.getenv("model")
        # )

    def evaluate(self, experiment_result: ExperimentResult) -> EvaluatorOutput:
        """Evaluate the experiment result using OpenAI's prompt-based evaluation."""
        # print('enter evaluation!!!\n')
        assert isinstance(self.config, AzureEvaluatorConfig)
        format_dict = copy.deepcopy(experiment_result.input_data.content)
        format_dict["raw_output"] = experiment_result.raw_output.text_output
        # print('raw output: ',format_dict["raw_output"])

        prompt = format_template(self.config.prompt, format_dict)
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        prompt[-1]["content"] += "\n\n" + CLASSIFY_STR.format(
            choices=choices_to_string(self.config.choices)
        )

        # Y Temp
        print('----------------------')
        print('\ncontent:  ',prompt[-1]["content"])

        # azure openai assistant, currently not working
        # response_content = create_assistant_and_get_response(
        #     prompt[-1]["content"], assistant_id=self.assistant_id
        # )
        response_content = client_response(prompt[-1]["content"])

        # Y Temp
        print("\nresponse: ",response_content)

        choice = extract_choice_from_response(
            response_content, self.config.choices
        )
        
        # Y Temp
        print('\nchoice:  ', choice)
        print('----------------------')

        score = calculate_choice_score(choice, self.config.choice_scores)
        return EvaluatorOutput(
            name=self.config.name,
            result=score if score is not None else choice,
            display_name=self.config.display_name,
            metric_calculators=self.config.metric_calculators
        )

    async def aevaluate(self, experiment_result: ExperimentResult) -> Any:
        print("enter async!!!\n")
        assert isinstance(self.config, AzureEvaluatorConfig)
        format_dict = copy.deepcopy(experiment_result.input_data.content)
        format_dict["raw_output"] = experiment_result.raw_output.text_output

        prompt = format_template(self.config.prompt, format_dict)
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        prompt[-1]["content"] += "\n\n" + CLASSIFY_STR.format(
            choices=choices_to_string(self.config.choices)
        )
        response = await acompletion_with_backpff(
            model="gpt-4",
            messages=prompt,
            temperature=0.5,
            n=1,
            max_tokens=1000,
            request_timeout=60,
        )
        # import pdb
        # pdb.set_trace()
        #response = openai.ChatCompletion.create(model="gpt-4", messages=prompt, temperature=0.5)
        response_content = response.choices[0].message.content
        choice = extract_choice_from_response(
            response_content, self.config.choices
        )
        score = calculate_choice_score(choice, self.config.choice_scores)
        return EvaluatorOutput(
            name=self.config.name,
            result=score if score is not None else choice,
            display_name=self.config.display_name,
            metric_calculators=self.config.metric_calculators
        )