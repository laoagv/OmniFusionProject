# -*- coding: utf-8 -*-
import asyncio
from websockets.asyncio.server import serve
import time
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
import json
import base64
from io import BytesIO

# from huggingface_hub import hf_hub_download

# Loading some sources of the projection adapter and image encoder
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')
from models import CLIPVisionTower

inspector = time

DEVICE = "cuda:0"
# DEVICE = "cpu"
PROMPT = "You are an assistant who perfectly describes images in detail and answers questions about them. When describing the image, please include details such as objects, colors, actions, and overall context. After providing a detailed description, be ready to answer specific questions related to the image. If a general question is asked similar to **Question:** What is in the picture?, then describe all the elements of the image. Now, describe the provided image and be prepared to answer any questions about it. Generate more than one sentence. Whats in the picture?"

tokenizer = AutoTokenizer.from_pretrained("./", subfolder="tokenizer", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("./", subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)

# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
projection = torch.load("projection.pt", map_location=DEVICE)
special_embs = torch.load("special_embeddings.pt", map_location=DEVICE)

clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.bfloat16)
 
startTime = time.time()

def gen_answer(model, tokenizer, clip, projection, query, special_embs, image=None):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 500,
        "early_stopping": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
        # "max_length": 300,
    }

    with torch.no_grad():
        print(time.time()-startTime)
        print(0)
        image_features = clip.image_processor(image, return_tensors='pt')
        
        image_embedding = clip(image_features['pixel_values']).to(device=DEVICE, dtype=torch.bfloat16)
        
        projected_vision_embeddings = projection(image_embedding).to(device=DEVICE, dtype=torch.bfloat16)
        
        prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        
        question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        
        prompt_embeddings = model.model.embed_tokens(prompt_ids).to(torch.bfloat16)
        
        question_embeddings = model.model.embed_tokens(question_ids).to(torch.bfloat16)
        
        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                projected_vision_embeddings,
                special_embs['EOI'][None, None, ...],
                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...]
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
        
        out = model.generate(inputs_embeds=embeddings, **gen_params)
        
    out = out[:, 1:]
    generated_texts = tokenizer.batch_decode(out)[0]

    print(time.time()-startTime)
    print(10)
    return generated_texts


async def echo(websocket):
    async for message in websocket:
        request = json.loads(message)

        question = request['text']

        image = request['image']
        if image.startswith("data:image/"):
            image = image.split(",")[1]
        decoded_bytes = base64.b64decode(image)
        byte_io = BytesIO(decoded_bytes)
        img = Image.open(byte_io)

        answer = gen_answer(
            model,
            tokenizer,
            clip,
            projection,
            query=question,
            special_embs=special_embs,
            image=img
        )
        await websocket.send(answer)

# async def echo(websocket):
#     async for message in websocket:
#         await websocket.send(message)

async def main():
    print("server run")
    async with serve(echo, "localhost", 8765):
        await asyncio.get_running_loop().create_future()  # run forever

asyncio.run(main())