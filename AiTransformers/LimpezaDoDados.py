arquivos = ["corpus.txt", "Corpus1.txt", "corpus2.txt"]
corpus_final = ""

for arquivo in arquivos:
    with open(arquivo, "r", encoding="utf-8") as f:
        texto = f.read()
    
    # Cada arquivo tem seu START e END
    start = texto.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    end = texto.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    texto_limpo = texto[start:end] if start != -1 and end != -1 else texto

    linhas = texto_limpo.splitlines()
    linhas = [linha.strip() for linha in linhas if linha.strip()]

    # Remove a linha do START, se existir
    if linhas[0].startswith("*** START"):
        linhas = linhas[1:]

    corpus_final += "\n".join(linhas) + "\n\n"  # separa livros com duas linhas

# Salvar o corpus final limpo
with open("corpus_clean.txt", "w", encoding="utf-8") as f:
    f.write(corpus_final)

print("corpus_clean.txt pronto com TODOS os livros limpos!")
