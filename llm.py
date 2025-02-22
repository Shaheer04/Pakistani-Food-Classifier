from huggingface_hub import InferenceClient
import os

hf_token = os.getenv("HF_TOKEN")


client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.3",
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
