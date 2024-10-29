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
PROMPT = "You are an assistant who perfectly describes images in detail and answers questions about them. When describing the image, please include details such as objects, colors, actions, and overall context. After providing a detailed description, be ready to answer specific questions related to the image. Here is an example of what I expect from you: **Image Description:** A bright sunny day with clear blue skies. In the foreground, there is a park bench under a large oak tree. The bench is painted green, and it has intricate carvings on its backrest. On the bench sits a young woman reading a book. She wears a white dress and has long brown hair flowing down her shoulders. Behind the bench, there’s a playground with swings and slides. Children can be seen playing happily. The grass around the area is lush and well-maintained, with small patches of wildflowers adding color to the scene. **Question:** What is the woman wearing? **Answer:** The woman is wearing a white dress. Now, describe the provided image and be prepared to answer any questions about it."

tokenizer = AutoTokenizer.from_pretrained("./", subfolder="tokenizer", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("./", subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)

# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
projection = torch.load("projection.pt", map_location=DEVICE)
special_embs = torch.load("special_embeddings.pt", map_location=DEVICE)

clip = CLIPVisionTower("openai/clip-vit-large-patch14-336")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.bfloat16)

def gen_answer(model, tokenizer, clip, projection, query, special_embs, image=None):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
        "do_sample": False,
        "max_new_tokens": 200,
        "early_stopping": False,
        "num_beams": 3,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": 2,
        "pad_token_id": 2,
        "forced_eos_token_id": 2,
        "use_cache": True,
        "no_repeat_ngram_size": 4,
        "bad_words_ids": bad_words_ids,
        "num_return_sequences": 1,
    }

    with torch.no_grad():
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

    return generated_texts

# # img_url = "https://i.pinimg.com/originals/32/c7/81/32c78115cb47fd4825e6907a83b7afff.jpg"
# question = "Кто изображен на картинке, что ты можешь сказать о нем?    "
# img = Image.open("test1.png")
img_urls = [
# "https://transport.mos.ru/common/upload/public/image/peshmetro_(1).jpg",
 # "https://udoba.org/sites/default/files/h5p/content/90357/images/image-6491c81a64cb7.png",
# "https://cdn.iportal.ru/preview/e1forum/photo/47ac5290b4b95aea5c48e4debd3ef9530c9c434a_700_700.jpg",
# "https://sun6-22.userapi.com/impg/YjtXKJddC8YFZUJqLqBMqeOD0VSgl7K5Sc1IkA/Z7WHZWyOaWI.jpg?size=604x365&quality=96&sign=31580b7776dd2a401722b63a38b095f3&type=album",
# "https://xakep.ru/wp-content/uploads/2017/11/142697/zerodaygraph.png",
"https://i.ytimg.com/vi/SHu3T9j70I4/maxresdefault.jpg",
# "https://www.sourcecodester.com/sites/default/files/2023-10/python-syntaxerror-continue-not-in-loop-1.png",
"https://images.squarespace-cdn.com/content/v1/581d0c7a15d5dbd666d2b128/1581721436816-Q0Y1KWV1445S8P781Y4M/Python_Iferror_Example.png",
"https://www.freecodecamp.org/news/content/images/2023/03/Screenshot-2023-03-13-at-17.58.33.png",
]
questions = ["найди ошибку в этом коде и исправь",
# "что за ошибка в коде и как ее исправить",
"найди ошибку в коде и скажи как ее исправить, укажи строку в которой находиться ошибка","найди ошибку в этом коде и исправь",
]
for imageUrl,question in zip(img_urls,questions):
    img = Image.open(urlopen(imageUrl)) #для открытия локальных изображений достаточно убрать urlopen

    answer = gen_answer(
        model,
        tokenizer,
        clip,
        projection,
        query=question,
        special_embs=special_embs,
        image=img
    )

    img.show()
    print(question)
    print(answer)