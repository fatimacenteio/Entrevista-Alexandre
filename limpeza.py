import spacy
import pandas as pd
import re

# Carregar modelo de língua portuguesa
nlp = spacy.load("pt_core_news_sm")

# Ler corpus.md
with open("corpus.md", "r", encoding="utf-8") as f:
    texto = f.read()

# Remover pontuação desnecessária e limpar
texto_limpo = re.sub(r"[^\w\sÀ-ÿ]", "", texto)  # remover pontuação
texto_limpo = texto_limpo.lower()  # tudo em minúsculas
texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()  # remover espaços extras

# Processar com spaCy
doc = nlp(texto_limpo)

# Extrair lemas e entidades
palavras = [token.text for token in doc if not token.is_stop and not token.is_punct]
lemas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
entidades = [(ent.text, ent.label_) for ent in doc.ents]

# Guardar entidades em ficheiro
with open("entidades_extraidas.txt", "w", encoding="utf-8") as f:
    for ent, tipo in entidades:
        f.write(f"{ent} ({tipo})\n")

# Criar dataset
df = pd.DataFrame({
    "palavra": palavras,
    "lema": lemas
})

# Contagem de frequência
frequencia = df["lema"].value_counts().reset_index()
frequencia.columns = ["lema", "frequencia"]

# Guardar dataset
df.to_csv("dataset.csv", index=False)
frequencia.to_csv("frequencia.csv", index=False)

print("✅ Limpeza e extração concluídas.")
