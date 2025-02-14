from huggingface_hub import InferenceClient
import os

hf_token = os.getenv("HF_TOKEN")


client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

def talk_to_llm(prompt):
    response_content = ""
    for message in client.chat_completion(
        messages=[
            {"role": "user", "content": prompt}],
        stream=True,
    ):
        response_content += message.choices[0].delta.content
    return response_content
