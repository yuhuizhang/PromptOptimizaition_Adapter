# written by Ruilin Cai

import os
import openai
from openai import AzureOpenAI

from yival.logger.token_logger import TokenLogger
from yival.schemas.experiment_config import MultimodalOutput
from yival.states.experiment_state import ExperimentState
from yival.wrappers.string_wrapper import StringWrapper
# from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

# [INPUT] edit the template for different cases, initial prompt
# template = """Generate a paragraph of text in Portuguese within 110 to 160 characters that related to {keyword}."""
# substitute by os key

def extract_variables(**kwargs):
    # Extract the state from the last positional argument
    state = kwargs['state']
    kwargs.pop('state')

    val = text_generation(kwargs, state)
    
    return val


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def text_generation(variable_dict: dict, state: ExperimentState) -> MultimodalOutput:
    '''generate prompt for chatgpt based on the input'''
    logger = TokenLogger()
    logger.reset()

    messages = [{
        "role":
        "system",
        "content":
        "You are a helpful assistant that generates required text content."
    }, {
        "role":
        "user",
        "content":
        str(
            StringWrapper(
                template=os.getenv("template"),
                variables=variable_dict,
                name="task",
                state=state
            )
        )
    }]

    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    response = client.chat.completions.create(
        max_tokens=int(os.getenv("max_tokens")),
        model=os.getenv("model"), # model = "deployment_name".
        messages=messages
    )

    # # old version openai==0.27.1
    # openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    # openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
    # openai.api_type = 'azure'
    # openai.api_version = '2023-12-01-preview' # this might change in the future

    # response = openai.ChatCompletion.create(
    #     engine='xxxx', # engine = "deployment_name".
    #     messages=messages,
    #     max_tokens=4000
    # )

    #  azure call based on litellm
    # deployment_name='xxxx' #This will correspond to the custom name you chose for your deployment when you deployed a model. 
    # response = completion(
    #     model = deployment_name, 
    #     messages = messages
    # )
    if not response:
        print('no response!!!!!!!! agent not working!!!!!')

    res = MultimodalOutput(
        text_output=response.choices[0].message.content,
    )
    token_usage = response.usage.total_tokens
    logger.log(token_usage)
    return res