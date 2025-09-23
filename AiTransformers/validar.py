import torch
print(torch.cuda.is_available())  # True = CUDA disponível
print(torch.cuda.get_device_name(0))  # Nome da GPU
