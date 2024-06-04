# written by Ruilin Cai

import os
import asyncio
import time

from typing import Dict, List, Union
from IPython.display import clear_output
from openai import AzureOpenAI, AsyncAzureOpenAI
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

# need to be carefully rewritten
# function for parallel agents running, not working now
async def parallel_completions(
    message_batches,
    model,
    max_tokens,
    temperature=1.3,
    presence_penalty=0,
    pbar=None,
    logit_bias=None
):
    """
    Asynchronous function to perform parallel completions using Azure OpenAI's API.

    Args:
        message_batches (list): A list containing batches of messages for
                                completion.
        model (str): The model to be used for completion.
        max_tokens (int): Maximum tokens to be used for completion.
        temperature (float, optional): Sampling temperature. Defaults to 1.3.
        presence_penalty (float, optional): Presence penalty for completion.
                                            Defaults to 0.
        pbar (optional): A progress bar instance to show progress. Defaults to
                        None.
        logit_bias (optional): Bias for the logit. Defaults to None.

    Returns:
        list: A list of responses containing completions for each message
            batch.
    """
    client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    tasks = [
        asyncio.ensure_future(
            client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens
            )
        ) for messages in message_batches
    ]

    responses = await asyncio.gather(*tasks)

    return responses


# ai assistant for evaluator
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

def create_assistant(name, tools, model, instructions):
    """
    Send a request to create an assistant with the specified parameters.
    
    :param name: Name of the assistant.
    :param tools: List of tools to be used by the assistant.
    :param model: Model version of the assistant.
    :param instructions: Role of the assistant.
    :param api_key: API key for authorization.
    :return: Response from the API.
    """
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        model=model, # replace with model deployment name. 
        tools=tools
    )

    # Return the response object
    return assistant.model_dump(mode="json")["id"]

def create_thread():
    """
    create thread
    """
    print("create thread\n")
    thread = client.beta.threads.create()

    # Return the response object
    return thread.model_dump(mode="json")["id"]

def post_message_to_thread(thread_id, content):
    """
    Send a message to a specific thread on the OpenAI API.

    :param thread_id: The identifier of the thread.
    :param content: The content of the message to be sent.
    :return: Response from the API.
    """
    print("post message to thread\n")
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )

    # Return the response object
    return message

def create_run(thread_id, assistant_id):
    """
    Start a run for an assistant in a specific thread on the OpenAI API.

    :param thread_id: The identifier of the thread.
    :param assistant_id: The identifier of the assistant.
    :return: Response from the API.
    """
    print("create run\n")
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        #instructions="New instructions" #You can optionally provide new instructions but these will override the default instructions
    )

    # Return the response object
    return run.model_dump(mode="json")["id"]

def check_run_status(thread_id, run_id) -> bool:
    """
    Check the status of a run within a thread on the OpenAI API.

    :param thread_id: The identifier of the thread.
    :param run_id: The identifier of the run.
    :return: True if the status is 'completed', False otherwise.
    """
    print("check run status\n")
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )

    status = run.status
    return status == "completed" or status == "failed" or status == "expired" or status == "cancelled"

def list_messages(thread_id):
    """
    List messages in a specific thread on the OpenAI API.

    :param thread_id: The identifier of the thread.
    :return: Response from the API.
    """
    print("list messages\n")
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )

    # Return the response object
    return messages.model_dump(mode="json")

def create_assistant_and_get_response(input_message, assistant_id):
    # Helper function to check run status and wait until it is completed
    
        # original code below:
        # while True:
        #     if check_run_status(thread_id, run_id):
        #         break  # Exit the loop if the run is completed
        #     time.sleep(
        #         1
        #     )  # Sleep for some time before checking again to avoid rate limiting

    # Reuse the previous functions, modifying them if necessary for the current context

    # Assuming create_thread(), post_message_to_thread(), create_run(), check_run_status(), and list_messages() are already defined

    # Start the workflow
    thread_id = create_thread()
    print(thread_id, '\n')
    post_message_to_thread(thread_id, input_message)
    run_id = create_run(thread_id, assistant_id)
    print(run_id, '\n')
    wait_for_run_completion(thread_id, run_id)  # Wait for the run to complete
    messages = list_messages(thread_id)  # Retrieve the list of messages
    return messages['data'][0]['content'][0]['text']['value']

def wait_for_run_completion(thread_id, run_id):

    print('start waiting...')
    run = client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )
    status = run.status
    print(f'Status: {status}')
    start_time = time.time()

    while status not in ["completed", "cancelled", "expired", "failed"]:
        time.sleep(5)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id,run_id=run_id)
        print("Elapsed time: {} minutes {} seconds".format(int((time.time() - start_time) // 60), int((time.time() - start_time) % 60)))
        status = run.status
        print(f'Status: {status}')
        clear_output(wait=True)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def client_response(input_message):

    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    response = client.chat.completions.create(
        model=os.getenv("model"), # model = "deployment_name".
        max_tokens=2000,
        messages= [{
            "role":
            "system",
            "content":
            "You are a helpful assistant that evaluate given text content following specific criteria."
        }, {
            "role":
            "user",
            "content":
            input_message
        }]
    )

    return response.choices[0].message.content
    

# next part
def azure_call(
    dialogues: Union[str, List[Dict[str, str]]],
    model_name=os.getenv("model")
) -> str:
    
    e_client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    response = e_client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=dialogues,
        temperature = 0.5

    )

    output_str = response.choices[0].message.content
    return output_str