
#First step quantization
from transformers import AutoModelForCausalLM
import safetensors.torch
import torch

# # Указываем путь до директории с файлами модели
# model_path = "tuned-model"
# DEVICE = "cuda:0"

# # Загружаем модель в формате safetensors
# # weights = safetensors.torch.load_file(model_path + "/model.safetensors")

# # Создаем объект модели
# # model = AutoModelForCausalLM.from_pretrained("OmniFusion", state_dict=weights)
# model = AutoModelForCausalLM.from_pretrained("./", subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map=DEVICE)

# # Сохраняем модель в PyTorch формате
# torch.save(model.state_dict(), "omnifusion_pytorch.pt")
import torch
from torch.quantization import quantize_dynamic

# # Загрузим модель из PyTorch файла
# # model = AutoModelForCausalLM.from_pretrained("Mistral-7B-v0.1").to("cuda:0")
# model = AutoModelForCausalLM.from_pretrained("./", subfolder="tuned-model", torch_dtype=torch.bfloat16, device_map="cpu")
# # state_dict = torch.load("omnifusion_pytorch.pt")
# # model.load_state_dict(state_dict)
# print("Модель загружена")
# # Применяем динамическую квантизацию
# quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# print("Квантизирована")

# # Сохраняем квантизированную модель обратно в PyTorch формате
# torch.save(quantized_model.state_dict(), "omnifusion_quantized.pt")
# print("Сохранена")

special_embeddings = torch.load("special_embeddings.pt")

# Применение динамической квантизации
quantized_special_embeddings = quantize_dynamic(special_embeddings, dtype=torch.qint8)

# Сохранение квантизированного специального эмбеддинга
torch.save(quantized_special_embeddings.state_dict(), "quantized_special_embeddings.pt")