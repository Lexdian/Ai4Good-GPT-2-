from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# 1️⃣ Carregar modelo e tokenizer fine-tunado
model_path = "./gpt2_sherlock"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Modo avaliação

# 2️⃣ Função para gerar texto e extrair informações
def gerar_texto_com_info(prompt, max_length=200, temperature=0.7, top_k=50, top_p=0.95):
    # Tokenizar input e mover para GPU
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Criar attention mask
    attention_mask = torch.ones(inputs.shape, device=device)

    # Forward pass para pegar logits, atenção e embeddings
    with torch.no_grad():
        outputs = model(inputs, output_attentions=True, output_hidden_states=True)
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        attentions = outputs.attentions  # lista: [num_layers][batch, heads, seq_len, seq_len]
        hidden_states = outputs.hidden_states  # lista: [num_layers+1][batch, seq_len, hidden_dim]

    # Próximos tokens mais prováveis (último token da sequência)
    probs = F.softmax(logits[0, -1], dim=-1)
    topk_tokens = torch.topk(probs, k=5)
    tokens_mais_provaveis = [(tokenizer.decode([i.item()]), p.item()) for i, p in zip(topk_tokens.indices, topk_tokens.values)]

    # Gerar texto
    generated = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    texto_gerado = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {
        "texto_gerado": texto_gerado,
        "tokens_mais_provaveis": tokens_mais_provaveis,
        "attentions": attentions,
        "hidden_states": hidden_states
    }

# 3️⃣ Testar com um prompt
prompt = "Sherlock Holmes entered the dimly lit room and immediately noticed:"
resultado = gerar_texto_com_info(prompt, max_length=200)

print("=== TEXTO GERADO ===")
print(resultado["texto_gerado"])
print("\n=== TOKENS MAIS PROVÁVEIS PARA O PRÓXIMO TOKEN ===")
for t, p in resultado["tokens_mais_provaveis"]:
    print(f"{t} -> {p:.4f}")
print("\n=== ATENÇÃO ===")
print(f"Número de camadas: {len(resultado['attentions'])}")
print(f"Shape da atenção da primeira camada: {resultado['attentions'][0].shape}")
print("\n=== EMBEDDINGS ===")
print(f"Número de camadas de hidden_states: {len(resultado['hidden_states'])}")
print(f"Shape do embedding da primeira camada: {resultado['hidden_states'][0].shape}")
