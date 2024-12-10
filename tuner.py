# -*- coding: utf-8 -*-
import time
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from urllib.request import urlopen
import torch.nn as nn
# from huggingface_hub import hf_hub_download

# Loading some sources of the projection adapter and image encoder
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="models.py", local_dir='./')
from models import CLIPVisionTower


DEVICE = "cuda:0"
# DEVICE = "cpu"
# PROMPT = "Вы помошник, который прекрасно и подробно описывает изображения и отвечает на вопросы по ним. При описании изображения укажите такие детали, как объекты, цвета, действия и общий контекст. После предоставления подробного описания будьте готовы ответить на конкретные вопросы, связанные с изображением. Если задан общий вопрос типа **Вопрос:** Что изображено на картинке?, то опишите все элементы изображения. Теперь опишите предоставленное изображение и будьте готовы ответить на любые вопросы о нем. Создайте более одного предложения."

tokenizer = AutoTokenizer.from_pretrained("./", subfolder="tokenizer", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("./", subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)

# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
projection = torch.load("projection.pt", map_location=DEVICE)
special_embs = torch.load("special_embeddings.pt", map_location=DEVICE)

clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.bfloat16)
 


def gen_answer(model, tokenizer, clip, projection, query, special_embs, params, prompt , image=None):
    torch.cuda.empty_cache()
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = params
    gen_params["bad_words_ids"] = bad_words_ids

    with torch.no_grad():

        image_features = clip.image_processor(image, return_tensors='pt')
        
        image_embedding = clip(image_features['pixel_values']).to(device=DEVICE, dtype=torch.bfloat16)
        
        projected_vision_embeddings = projection(image_embedding).to(device=DEVICE, dtype=torch.bfloat16)

        print(image_embedding.shape)
        print(projected_vision_embeddings.shape)
        
        prompt_ids = tokenizer.encode(f"{prompt}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)
        
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
    # generated_texts = tokenizer.batch_decode(out)[:6]

    return generated_texts
firstTime = time.time()
count = 0 
prompts = [
# "This is a dialog with AI assistant.\n",
# "You are an assistant who perfectly describes images.",
# "Вы помошник, который прекрасно описывает картинки и отвечаетна вопросы по ним",
# "Ты отлчино находишь текст на картинках и отвечаешь на вопросы по нему",
# "Твоя задача находить на изображениях слова ГОСТ и определять их номер, который идет сразу после них",
"Проанализируй предоставленное изображение на наличие конфиденциальной информации (например, пароли, логины, документы с чувствительными данными). Определи любые потенциальные риски.",
"На изображении скриншот экрана. Определи, есть ли на нём открытые окна с конфиденциальной информацией (e.g., почта, финансовые данные, панель администратора) или уязвимые элементы.",
"Если на изображении есть текст, проанализируй его содержимое и укажи, есть ли упоминания о конфиденциальной информации или уязвимостях.",
# "",
# "",
# "",
# "",
# "",
]
question="Опиши изображение подробно"
# image="test1.jpeg"
images = ["test1.jpeg", "atack.jpg"]
with open('anotherPrompts.txt','w+',encoding="utf-8") as logs:
    for image in images:
        for PROMPT in prompts:
            for do_sample,max_new_tokens,early_stopping,num_beams in zip([False, False],[350, 650],[False, True],[1, 1]):
                count+=1
                startTime = time.time()- firstTime
                params = {
                    "do_sample": do_sample,
                    "max_new_tokens": max_new_tokens,
                    "early_stopping": early_stopping,
                    "num_beams": num_beams,
                    "repetition_penalty": 1.0,
                    "remove_invalid_values": True,
                    "eos_token_id": 2,
                    "pad_token_id": 2,
                    "forced_eos_token_id": 2,
                    "use_cache": True,
                    "no_repeat_ngram_size": 4,
                    "num_return_sequences": 1,
                    # "max_length": 300,
                }
                img = Image.open(image)
                answer = gen_answer(
                    model,
                    tokenizer,
                    clip,
                    projection,
                    query=question,
                    special_embs=special_embs,
                    params=params,
                    prompt=PROMPT,
                    image=img,
                )
                endTime = time.time()- firstTime
                logs.write("do_sample: "+str(do_sample)+" early_stopping: "+str(early_stopping)+" max_new_tokens: "+str(max_new_tokens)+" num_beams: "+str(num_beams)+"\n")
                logs.write("Start time: "+str(startTime)+" End time: "+str(endTime)+" Total time: "+str(endTime - startTime)+"\n")
                logs.write("Question: "+question+"\n")
                logs.write("PROMPT: "+PROMPT+"\n")
                logs.write("Answer: "+answer+"\n")
                print("do_sample: "+str(do_sample)+" early_stopping: "+str(early_stopping)+" max_new_tokens: "+str(max_new_tokens)+" num_beams: "+str(num_beams))
                print("Start time: "+str(startTime)+" End time: "+str(endTime)+" Total time: "+str(endTime - startTime))
                print("Question: "+question)
                print("PROMPT: "+PROMPT)
                print("Answer: "+answer)
                print(str(count)+"/5")




# with open('logs.txt','w+',encoding="utf-8") as logs:
#     for do_sample in [True,False]:
#         for early_stopping in [True,False]:
#             for max_new_tokens in [50,200,500]:
#                 for num_beams in [1,4,7]:
#                     if early_stopping==True and num_beams==1:
#                         continue
#                     count+=1
#                     startTime = time.time()- firstTime
#                     params = {
#                         "do_sample": do_sample,
#                         "max_new_tokens": max_new_tokens,
#                         "early_stopping": early_stopping,
#                         "num_beams": num_beams,
#                         "repetition_penalty": 1.0,
#                         "remove_invalid_values": True,
#                         "eos_token_id": 2,
#                         "pad_token_id": 2,
#                         "forced_eos_token_id": 2,
#                         "use_cache": True,
#                         "no_repeat_ngram_size": 4,
#                         "num_return_sequences": 1,
#                         # "max_length": 300,
#                     }
#                     img = Image.open(image)
#                     answer = gen_answer(
#                         model,
#                         tokenizer,
#                         clip,
#                         projection,
#                         query=question,
#                         special_embs=special_embs,
#                         params=params,
#                         image=img,
#                     )
#                     endTime = time.time()- firstTime
#                     logs.write("do_sample: "+str(do_sample)+" early_stopping: "+str(early_stopping)+" max_new_tokens: "+str(max_new_tokens)+" num_beams: "+str(num_beams)+"\n")
#                     logs.write("Start time: "+str(startTime)+" End time: "+str(endTime)+" Total time: "+str(endTime - startTime)+"\n")
#                     logs.write("Question: "+question+"\n")
#                     logs.write("Answer: "+answer+"\n")
#                     print("do_sample: "+str(do_sample)+" early_stopping: "+str(early_stopping)+" max_new_tokens: "+str(max_new_tokens)+" num_beams: "+str(num_beams))
#                     print("Start time: "+str(startTime)+" End time: "+str(endTime)+" Total time: "+str(endTime - startTime))
#                     print("Question: "+question)
#                     print("Answer: "+answer)
#                     print(str(count)+"/36")