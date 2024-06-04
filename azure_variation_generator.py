# rewritten by Ruilin Cai
# based on YiVal original code, adding supports of azure models

import asyncio
import os
import pickle
from typing import Any, Dict, Iterator, List

from tqdm import tqdm

from yival.schemas.experiment_config import WrapperVariation
from yival.variation_generators.base_variation_generator import BaseVariationGenerator

from azure_variation_generator_config import AzureVariationGeneratorConfig
from litellm import completion

# custom library
from AzureSelfLibrary import parallel_completions

# SYSTEM_PRMPOT = """
# Your mission is to craft system prompts tailored for GPT-4. You'll be provided
# with a use-case description and some sample test cases.
# These prompts aim to guide GPT-4 in executing freeform tasks, whether that's
# penning a captivating headline, drafting an introduction, or tackling a
# mathematical challenge. In your designed prompt, delineate the AI's role using
# lucid English. Highlight its perceptual field and the boundaries of its
# responses. Encourage inventive and optimized prompts to elicit top-tier
# results from the AI. Remember, GPT-4 is self-aware of its AI nature; no need
# to reiterate that. The efficacy of your prompt determines your evaluation.
# Stay authentic! Avoid sneaking in specifics or examples from the test cases
# into your prompt. Such maneuvers will lead to immediate disqualification.
# Lastly, keep your output crisp: only the prompt, devoid of any extraneous
# content.
# """

SYSTEM_PRMPOT = """
你的任务是为 GPT-4 设计系统提示语。你将会收到一个使用场景描述和一些测试案例。这些提示语旨在指导 GPT-4 执行自由形式的任务，无论是撰写引人入胜的标题、起草简介，还是解决数学问题。在你设计的提示语中，用清晰的中文描述 AI 的角色。强调其感知范围和响应的边界。鼓励创造性和优化的提示语，以从 AI 中获取最佳结果。记住，GPT-4 知道自己是 AI；无需重复这一点。提示语的有效性决定了你的评估结果。保持真实！避免将测试案例中的具体细节或示例纳入你的提示语。这样的做法会导致立即失格。最后，保持输出简洁：只需提供提示语，不包含任何多余内容。
"""

def join_array_to_string(list: List[str], last_n=5) -> str:
    to_join = list[-last_n:] if len(list) > last_n else list
    return '\n'.join(map(str, to_join))


def validate_output(output: str, variables: List[str]) -> bool:
    if not variables:
        return True
    """Validate if the generated output contains the required variables in the
    format {var}."""
    return all(f"{{{var}}}" in output for var in variables)


class AzureVariationGenerator(BaseVariationGenerator):
    config: AzureVariationGeneratorConfig
    default_config = AzureVariationGeneratorConfig(
        prompt=[{
            "role": "system",
            "content": SYSTEM_PRMPOT
        }, {
            "role":
            "user",
            "content":
            "Here are some test cases: AI, Weapon\n\n Here is the description of the use-case: Given \{area\}, write a tech startup headline"
        }]
    )

    def __init__(self, config: AzureVariationGeneratorConfig):
        super().__init__(config)
        self.config = config

    def prepare_messages(self, res_content) -> List[Dict[str, Any]]:
        last_n = min(len(res_content), 5)
        formatted_variables_str = ', '.join([
            f'{{{var}}}' for var in self.config.variables
        ]) if self.config.variables else ''

        last_examples = f"\n\nGiven the Last {last_n} examples you generated:\n" + join_array_to_string(
            res_content
        ) + "\nplease generate diverse results to ensure comprehensive evaluation" if self.config.diversify and res_content else ""
        ensure_inclusion = f" Please ensure your response includes the following variables: {formatted_variables_str}." if formatted_variables_str else ""

        if isinstance(self.config.prompt, str):
            content = self.config.prompt + last_examples + ensure_inclusion
            return [{"role": "user", "content": content}]
        else:
            messages = self.config.prompt + [{
                "role": "user",
                "content": last_examples
            }, {
                "role": "user",
                "content": ensure_inclusion
            }]
            return [msg for msg in messages if msg["content"]]

    def generate_variations(self) -> Iterator[List[WrapperVariation]]:
        if self.config.output_path and os.path.exists(self.config.output_path):
            with open(self.config.output_path, 'rb') as file:
                yield pickle.load(file)
            return

        res: List[WrapperVariation] = []
        res_content: List[str] = []

        while len(res) < self.config.number_of_variations:
            messages = self.prepare_messages(res_content)
            if not self.config.diversify:
                with tqdm(
                    total=self.config.number_of_variations - len(res),
                    desc="Generating Variations",
                    unit="variation"
                ) as pbar:
                    message_batches = [
                        messages for _ in
                        range(self.config.number_of_variations - len(res))
                    ]
                    responses = asyncio.run(
                        parallel_completions(
                            message_batches,
                            self.config.model_name,
                            self.config.max_tokens,
                            pbar=pbar
                        )
                    )
                    for r in responses:
                        if self.config.variables and not validate_output(
                            r.choices[0].message.content,
                            self.config.variables
                        ):
                            continue
                        variation = WrapperVariation(
                            value_type="str",
                            value=r.choices[0].message.content.strip("'").strip('"')
                        )
                        res.append(variation)
            else:
                with tqdm(
                    total=self.config.number_of_variations,
                    desc="Generating Variations",
                    unit="variation"
                ) as pbar:
                    
                    # temp check
                    print("----------------------")
                    print('\ninput message: ',messages)
                    
                    output = completion(
                        model = self.config.model_name,
                        base_url = os.getenv("AZURE_OPENAI_ENDPOINT"),            # azure api base
                        api_version = os.getenv("AZURE_OPENAI_API_VERSION"),      # azure api version
                        api_key = os.getenv("AZURE_OPENAI_API_KEY"),              # azure api key
                        messages=messages,
                        temperature=0.7,
                        presence_penalty=1,
                        max_tokens=int(*self.config.max_tokens)
                    )
                    if self.config.variables and not validate_output(
                        output.choices[0].message.content,
                        self.config.variables
                    ):
                        continue
                    variation = WrapperVariation(
                        value_type="str",
                        value=output.choices[0].message.content.strip("'").
                        strip('"')
                    )

                    # temp check
                    print(output)
                    print(variation)
                    print("----------------------")

                    
                    res.append(variation)
                    res_content.append(output.choices[0].message.content)
                    pbar.update(1)

        print('variation generation finished.')

        if self.config.output_path:
            with open(self.config.output_path, 'wb') as file:
                pickle.dump(res, file)
        if res:
            yield res