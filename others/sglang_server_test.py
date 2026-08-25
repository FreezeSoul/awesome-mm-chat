import openai
from transformers import AutoTokenizer


def demo1():
    client = openai.Client(base_url=f"http://127.0.0.1:5678/v1", api_key="None")

    response = client.chat.completions.create(
        model="qwen/qwen3-8b-instruct",
        messages=[
            {"role": "user", "content": "List 3 countries and their capitals."},
        ],
        temperature=0,
        max_tokens=64,
    )
    print(response)


def demo2():
    client = openai.Client(base_url=f"http://127.0.0.1:5678/v1", api_key="None")

    response = client.chat.completions.create(
        model="internlm/Intern-S1-mini ", # internlm/Intern-S1-mini  qwen/qwen3-8b-instruct
        messages=[{"content": [{"image_url": {"image_wh": [404, 162],
                                              "url": "/mnt/shared-storage-user/llmrazor-share/data/geometry3k/images/test_2.jpg"},
                                "type": "image_url"}, {
                                   "text": "Find the area of the figure to the nearest tenth. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.",
                                   "type": "text"}], "role": "user"}],
        temperature=0,
        max_tokens=64,
    )
    print(response)


def demo3():
    messages = [{"content": [{"image_url": {"image_wh": [404, 162],
                                            "url": "/mnt/shared-storage-user/llmrazor-share/data/geometry3k/images/test_2.jpg"},
                              "type": "image_url"}, {
                                 "text": "<IMG_CONTEXT>Find the area of the figure to the nearest tenth. You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.",
                                 "type": "text"}], "role": "user"}]
    # model_path = '/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472'
    model_path = '/mnt/shared-storage-user/llmrazor-share/model/intern-s1-mini-hha-fix_tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenized_messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_token_ids = tokenizer(tokenized_messages, add_special_tokens=False)["input_ids"]
    # tokenized_messages=tokenized_messages.replace('<|vision_start|><|image_pad|><|vision_end|>','')
    print(tokenized_messages)
    print(prompt_token_ids)
    # ids = tokenizer('<image>')['input_ids']
    # print(ids)
    import requests

    response = requests.post(
        "http://127.0.0.1:5678/generate",
        json={
            # "text": tokenized_messages,
            'input_ids': prompt_token_ids,
            "image_data": "/mnt/shared-storage-user/llmrazor-share/data/geometry3k/images/test_2.jpg",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 64,
            },
        },
    )
    print(response.json())

if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
