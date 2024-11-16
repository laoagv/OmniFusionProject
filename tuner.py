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
"This is a dialog with AI assistant.\n",
"You are an assistant who perfectly describes images.",
"Вы помошник, который прекрасно описывает картинки и отвечаетна вопросы по ним",
"You are an assistant who perfectly describes images in detail and answers questions about them. When describing the image, please include details such as objects, colors, actions, and overall context. After providing a detailed description, be ready to answer specific questions related to the image. If a general question is asked similar to **Question:** What is in the picture?, then describe all the elements of the image. Now, describe the provided image and be prepared to answer any questions about it. Generate more than one sentence.",
"Вы помошник, который прекрасно и подробно описывает изображения и отвечает на вопросы по ним. При описании изображения укажите такие детали, как объекты, цвета, действия и общий контекст. После предоставления подробного описания будьте готовы ответить на конкретные вопросы, связанные с изображением. Если задан общий вопрос типа **Вопрос:** Что изображено на картинке?, то опишите все элементы изображения. Теперь опишите предоставленное изображение и будьте готовы ответить на любые вопросы о нем. Создайте более одного предложения.",
"You are an assistant who perfectly describes images in detail and answers questions about them. When describing the image, please include details such as objects, colors, actions, and overall context. After providing a detailed description, be ready to answer specific questions related to the image. Here is an example of what I expect from you: **Image Description:** A bright sunny day with clear blue skies. In the foreground, there is a park bench under a large oak tree. The bench is painted green, and it has intricate carvings on its backrest. On the bench sits a young woman reading a book. She wears a white dress and has long brown hair flowing down her shoulders. Behind the bench, there’s a playground with swings and slides. Children can be seen playing happily. The grass around the area is lush and well-maintained, with small patches of wildflowers adding color to the scene. **Question:** What is the woman wearing? **Answer:** The woman is wearing a white dress. Now, describe the provided image and be prepared to answer any questions about it.",
"Вы — эксперт по анализу изображений и визуальному контенту. Ваша задача — предоставлять подробные описания картинок и отвечать на вопросы о них. Вы должны учитывать все детали, включая контекст, объекты, действия, эмоции и любые другие важные аспекты. Ваш ответ должен быть максимально информативным и точным. Также важно поддерживать разговорный стиль общения, делая объяснения понятными и доступными для широкой аудитории. Вы можете работать как с русскими, так и с английскими изображениями и вопросами. Формат взаимодействия следующий: 1. Я предоставляю изображение и/или задаю вопрос. 2. Вы даете подробное описание изображения и/или отвечаете на мой вопрос. 3. Если нужно, я могу задать дополнительные вопросы или предоставить новую картинку. Пожалуйста, начните с описания изображения.",
"You are an advanced visual assistant specializing in detailed descriptions of images and answering complex questions about their content. When describing an image, your task is to provide a comprehensive and structured analysis that includes: General setting and atmosphere Key objects and their characteristics (colors, shapes, textures) Actions or interactions between elements Contextual information that helps understand the scene After completing the description, you should be prepared to answer specific questions about the image, demonstrating deep understanding of its content. Here is an example of the level of detail expected from you: Image Description: A bright summer day with clear blue skies. In the center of the image is a serene lake surrounded by lush greenery. The water reflects the sky, creating a mirror-like effect. Near the shoreline, several ducks swim peacefully, their feathers glistening in the sunlight. On the left side, a wooden dock extends into the water, with two people fishing. They wear casual clothing and have fishing rods in their hands. In the background, tall trees line the edge of the lake, providing shade and a sense of tranquility. Small waves gently lap against the shore, adding to the peaceful ambiance. Question: What are the people on the dock doing? Answer: The people on the dock are fishing. Now, based on this example, please provide a detailed description of the given image and be prepared for any follow-up questions.",
]
question="Выдели все объекты на изображении и перечисли их, какие потенциальные уязвимости присутствуют в данной системе?"
image="attackHighQual.png"

with open('logsPrompts.txt','w+',encoding="utf-8") as logs:
    for PROMPT in prompts:
        for do_sample,max_new_tokens,early_stopping,num_beams in zip([True,True,True,False],[200,500,500,500],[False,False,True,False],[1,1,4,1]):
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
            print(str(count)+"/32")




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