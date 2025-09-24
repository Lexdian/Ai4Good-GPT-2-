from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 1️⃣ Carregar modelo e tokenizer fine-tunado
model_path = "./gpt2_sherlock"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Detectar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 2️⃣ Função para gerar texto
def gerar_texto(prompt, max_length=200, temperature=0.7):
    # Tokenizar input e mover para GPU
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Gerar texto
    outputs = model.generate(
        inputs,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    
    # Decodificar output
    texto_gerado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return texto_gerado

# 3️⃣ Testar com um prompt
prompt = "Sherlock Holmes entered the dimly lit room and immediately noticed:"
texto = gerar_texto(prompt, max_length=200)
print(texto)

