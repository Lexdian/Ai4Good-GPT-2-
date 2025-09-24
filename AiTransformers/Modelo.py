from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1️⃣ Tokenizer e modelo pré-treinado (versão "medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")

# 2️⃣ Dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="corpus_clean.txt",
    block_size=512,  # tamanho dos pedaços de texto que o modelo processa
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 não usa masked LM
)

# 3️⃣ Configuração do treino
training_args = TrainingArguments(
    output_dir="./gpt2_medium_sherlock",
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=4,  # pode precisar reduzir se a GPU não tiver muita VRAM
    save_steps=200,
    save_total_limit=2,
    logging_steps=100,
    fp16=True,  # mixed precision
    learning_rate=5e-5,
)


# 4️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# 5️⃣ Treino
trainer.train()

# 6️⃣ Salvar modelo ajustado
model.save_pretrained("./gpt2_medium_sherlock")
tokenizer.save_pretrained("./gpt2_medium_sherlock")

print("Treino finalizado")
