import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import re
import os

# --- O TEXTO DA SUA ENTREVISTA ---
# Este é o texto completo da entrevista fornecido por você.
texto_entrevista_original = """
Boa tarde. Neste momento, inicio a minha entrevista com o meu colega Alexandre.
Que palavras usarias para descrever a tua infância?
Divertida, única, empolgante, inesquecível. Acho que "inesquecível" é a palavra-chave.
Onde cresceste e como era o sítio onde vivias?
Cresci na minha freguesia, onde ainda vivo atualmente, em Nino. Foi lá que fiz muitos dos meus amigos de infância, de juventude — amigos que ainda fazem parte da minha vida. Todas as minhas memórias de infância, pré-adolescência e adolescência são de lá.
Tens alguma memória especial da tua infância que te tenha marcado?
Tenho várias. Por exemplo, quando íamos à catequese, jogávamos sempre depois — esconde-esconde, jogo da macaca, lencinho. São memórias que levarei comigo. Jogar à bola com o McLaren, as primeiras saídas só com os amigos, festas de aniversário, Carnaval, Natal, Páscoa — tudo isso me marcou muito.
Qual era a tua brincadeira preferida quando eras criança?
Adorava jogar à apanhada e aos "pontos-pontos". Sempre que nos juntávamos, fosse durante a semana ou nas férias, íamos para o parque e jogávamos. Era uma rotina.
Havia algum brinquedo ou objeto que fosse especial para ti?
Não havia um brinquedo específico, mas guardo com carinho algumas peças de vestuário do tempo em que jogava futebol. T-shirts com o meu nome, chuteiras, casacos... evocam boas memórias da infância.
Como foi a tua experiência na escola primária? Tinhas alguma disciplina preferida?
Sempre gostei muito de inglês. Mesmo quando não entendia os filmes, sentia vontade de compreender. Também gostava de história e, claro, de futebol.
Lembras-te do teu primeiro amigo de infância? Ainda manténs contacto com ele?
Sim, claro. É filho da melhor amiga da minha mãe. Crescemos juntos. Dormíamos em casa um do outro, passámos fins de semana e férias juntos. Temos uma amizade sólida e mantemos contacto até hoje.
Qual era a tua rotina diária durante a infância?
Acordava cedo, sempre bem disposto. Passava os dias entre casa, o infantário e, quando a minha avó estava em Portugal, ficava com ela. A rotina era simples, mas marcante.
Tinhas alguma tradição familiar especial de que te lembres com carinho?
Hoje em dia temos uma tradição: aos domingos, o meu avô e o meu tio materno almoçam lá em casa. Desde pequeno que comemoramos o Natal e a Páscoa sempre juntos, na nossa casa. Isso sempre foi uma constante.
Havia alguma comida que adoravas ou detestavas quando eras criança?
Detestava pepino. Quando íamos ao McDonald's, pedia logo à minha mãe para tirar o pepino. Hoje é um dos meus vegetais favoritos. Ainda não consigo comer gordura da carne, mas como quase tudo hoje em dia.
O que desejavas ser quando eras criança?
Queria ser médico, para curar os diabetes da minha avó paterna. Mais tarde percebi que não tem cura, mas essa era a minha motivação. Também quis ser polícia, influenciado por séries americanas.
Se pudesses reviver um dia da tua infância, qual escolherias e porquê?
Um dia de Natal, com a minha avó materna e o meu avô paterno ainda vivos. Estarmos todos juntos na mesma mesa seria o dia perfeito para reviver. Não hesitaria.
Fim da entrevista.
"""

# --- LIMPEZA AVANÇADA DO TEXTO ---
partes = texto_entrevista_original.split("Fim da entrevista.")
texto_base = partes[0].strip()
texto_limpo_regex = re.sub(r'(?i)Pergunta:\s*|Resposta:\s*', '', texto_base).strip()
texto_entrevista_processado = "\n".join([linha for linha in texto_limpo_regex.split('\n') if linha.strip()]).strip()

if not texto_entrevista_processado:
    print("Erro: A variável 'texto_entrevista_processado' está vazia ou ficou vazia após a limpeza. Por favor, verifique o texto original.")
    exit()

# --- Configuração do diretório de saída para os gráficos ---
output_dir = 'graficos_analise'
os.makedirs(output_dir, exist_ok=True) # Cria a pasta se ela não existir

# --- Carregar o modelo SpaCy para português ---
print("Carregando modelo SpaCy para processamento de texto...")
nlp = None
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    print("Modelo SpaCy 'pt_core_news_sm' não encontrado. Tentando download automático...")
    try:
        spacy.cli.download("pt_core_news_sm")
        nlp = spacy.load("pt_core_news_sm")
        print("Download e carregamento do modelo SpaCy bem-sucedidos.")
    except Exception as e:
        print(f"Erro ao baixar ou carregar o modelo SpaCy: {e}")
        print("Certifique-se de que tem conectividade à internet e que o SpaCy está corretamente instalado.")
        print("Pode tentar 'python -m spacy download pt_core_news_sm' manualmente no terminal.")
        print("A análise de entidades será desativada.")
        exit()

# Processar o texto com SpaCy
doc = nlp(texto_entrevista_processado)

# --- Define a lista de palavras comuns a serem EXCLUÍDAS da análise de entidades ---
common_words_to_exclude = {
    "tua", "minha", "minhas", "teu", "meu", "meus",
    "eu", "tu", "ele", "ela", "nós", "você", "vocês",
    "que", "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "com", "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos", "para", "por", "e", "ou",
    "foi", "era", "eram", "tinha", "tinhas", "tenho", "cresci", "acho", "adorava", "queria", "percebi", "quis", "pudesses", "escolherias",
    "divertida", "única", "empolgante", "inesquecível", "especial", "preferida", "sólida", "diária", "simples", "marcante", "disposto", "materno", "paterna", "paterno", "favoritos", "próprio", "perfeito", "juntos", "todos", "primeiro", "outro", "primeiras", "melhor", "constante", "pouco", "quase", "muito",
    "lá", "sempre", "ainda", "hoje", "cedo", "bem",
    "boa", "tarde", "neste", "momento", "inicio", "entrevista", "colega",
    "sim", "claro", "não", "tudo", "isso", "várias", "alguma", "qual", "onde", "como", "quem", "o que", "se",
    "casa", "vida", "memórias", "parte", "rotina", "tempo", "brinquedo", "objeto", "peças", "vestuário",
    "chuteiras", "casacos", "filmes", "vontade",
    "amigo", "amiga", "mãe", "avó", "avô", "tio", "familiar", "familiares",
    "domingos", "infantário", "portugal", "mesa", "dia",
    "curar", "diabetes", "motivação", "séries", "americanas", "reviver",
    "freguesia", "juventude", "adolescência", "pré-adolescência", "bola", "saídas", "festas", "aniversário", "semana", "férias",
    "comida", "pepino", "vegetais", "gordura", "carne",
    "esconde-esconde", "jogo da macaca", "lencinho", "à apanhada", "pontos-pontos", # Brincadeiras
    "médico", "polícia", # Profissões
    "futebol", "história", "parque", "disciplina", # Desportos, disciplinas gerais, locais comuns
    "guardo", "lembras-te", "manténs", "acordava"
}
excluded_common_words_set = set(common_words_to_exclude)

# --- Extração e Análise de Entidades Nomeadas (Gráfico de Barras HORIZONTAL) ---
print("\n--- Analisando Entidades Nomeadas ---")
entidades_para_contagem = []
entidades_para_exibicao_map = {}
preposicoes_a_limpar_inicio = {"em", "de", "da", "do", "das", "dos", "para", "a", "à", "ao", "aos", "às", "com", "por"}

for ent in doc.ents:
    if ent.label_ in ["PER", "LOC", "ORG", "MISC"]:
        ent_text_stripped = ent.text.strip()
        
        found_prep_in_this_ent = False
        for prep in preposicoes_a_limpar_inicio:
            if ent_text_stripped.lower().startswith(prep + " "):
                ent_text_stripped = ent_text_stripped[len(prep) + 1:].strip()
                found_prep_in_this_ent = True
                break
        
        if not ent_text_stripped or len(ent_text_stripped) <= 1: 
             continue

        ent_text_lower = ent_text_stripped.lower()
        
        if ent_text_lower in excluded_common_words_set:
            continue
        
        entidades_para_contagem.append(ent_text_lower)

        if ent_text_lower not in entidades_para_exibicao_map:
            entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped
        elif ent_text_stripped[0].isupper() and not entidades_para_exibicao_map[ent_text_lower][0].isupper():
            entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped
        elif ' ' in ent_text_stripped and (ent_text_stripped.istitle() or ent_text_stripped[0].isupper()):
             entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped

termos_a_capitalizar_corretamente = {
    "alexandre": "Alexandre", "mclaren": "McLaren", "nino": "Nino", "natal": "Natal",
    "páscoa": "Páscoa", "carnaval": "Carnaval", "inglês": "Inglês", "mcdonald's": "McDonald's",
    "catequese": "Catequese", "história": "História", "portugal": "Portugal"
}

for key_lower, correct_capitalization in termos_a_capitalizar_corretamente.items():
    if key_lower in entidades_para_exibicao_map:
        entidades_para_exibicao_map[key_lower] = correct_capitalization

frequencia_entidades_lower = Counter(entidades_para_contagem)
df_entidades_data = []
for ent_lower, freq in frequencia_entidades_lower.most_common(10): # Top 10 entidades
    display_name = entidades_para_exibicao_map.get(ent_lower, ent_lower.capitalize())
    df_entidades_data.append([display_name, freq])
df_entidades = pd.DataFrame(df_entidades_data, columns=['Entidade', 'Frequência'])

if df_entidades.empty:
    print("Nenhuma entidade nomeada relevante encontrada no texto para gerar o gráfico após a filtragem.")
else:
    print("\nTabela de Entidades Nomeadas e Frequência:")
    print(df_entidades)

    plt.figure(figsize=(12, 8)) # Ajustado para possivelmente mais largura
    colors = plt.cm.viridis(df_entidades['Frequência'] / df_entidades['Frequência'].max())
    # Alterado para plt.barh (horizontal)
    plt.barh(df_entidades['Entidade'], df_entidades['Frequência'], color=colors)
    plt.xlabel('Frequência', fontsize=12) # X-label agora é Frequência
    plt.ylabel('Entidade', fontsize=12)   # Y-label agora é Entidade
    plt.title('Extração de Entidades Nomeadas', fontsize=14)
    plt.gca().invert_yaxis() # Inverte o eixo Y para que o mais frequente fique no topo
    plt.grid(axis='x', linestyle='--', alpha=0.7) # Grade no eixo X
    plt.tight_layout()
    output_filepath_entities = os.path.join(output_dir, 'grafico_entidades_nomeadas_horizontal.png') # Novo nome
    plt.savefig(output_filepath_entities, dpi=300, bbox_inches='tight')
    print(f"\nGráfico de entidades nomeadas (horizontal) salvo em: {output_filepath_entities}")
    plt.show()
    print("Geração do gráfico de entidades concluída.")


# --- Análise de Lema (Gráfico de Barras HORIZONTAL - Top 10 Lemas) ---
print("\n--- Analisando Lemas ---")

lemmas_to_exclude = set(nlp.Defaults.stop_words)
lemmas_to_exclude.update([
    "ser", "ter", "ir", "estar", "fazer", "vir", "poder", "querer", "saber", "achar", "dar", "dizer", "ver",
    "coisa", "dia", "tudo", "muito", "pouco", "assim", "bom", "mau", "mais", "menos",
    "a", "o", "e", "u", "de", "do", "da", "em", "um", "uma",
    "minha", "meu", "teu", "tua", "nosso", "nossa", "seu", "sua",
    "este", "esse", "aquele", "isto", "isso", "aquilo",
    "claro", "sim", "não", "bem", "mal", "só", "ainda", "já", "agora", "quando", "onde", "como", "qual", "que", "quem", "por", "para", "com", "mas", "se", "ou",
    "gente", "casa", "vida", "parte", "rotina", "tempo", "amigo", "memória", "tradição", "avó", "avô", "tio", "família", "comida", "filho", "disciplina", "futebol", "parque", "escola"
])

lemmas = []
for token in doc:
    if token.is_alpha and not token.is_stop and not token.is_punct and len(token.lemma_) > 2:
        lemma = token.lemma_.lower()
        if lemma not in lemmas_to_exclude:
            lemmas.append(lemma)

frequencia_lemmas = Counter(lemmas)
df_lemmas_data = []
for lemma, freq in frequencia_lemmas.most_common(10): # Top 10 lemas
    df_lemmas_data.append([lemma.capitalize(), freq])
df_lemmas = pd.DataFrame(df_lemmas_data, columns=['Lema', 'Frequência'])

if df_lemmas.empty:
    print("Nenhum lema relevante encontrado no texto para gerar o gráfico após a filtragem.")
else:
    print("\nTabela de Lemas Mais Frequentes:")
    print(df_lemmas)

    plt.figure(figsize=(12, 8)) # Ajustado para possivelmente mais largura
    colors_lemmas = plt.cm.plasma(df_lemmas['Frequência'] / df_lemmas['Frequência'].max())
    # Alterado para plt.barh (horizontal)
    plt.barh(df_lemmas['Lema'], df_lemmas['Frequência'], color=colors_lemmas)
    plt.xlabel('Frequência', fontsize=12) # X-label agora é Frequência
    plt.ylabel('Lema', fontsize=12)       # Y-label agora é Lema
    plt.title('10 Lemas Mais Frequentes', fontsize=14)
    plt.gca().invert_yaxis() # Inverte o eixo Y para que o mais frequente fique no topo
    plt.grid(axis='x', linestyle='--', alpha=0.7) # Grade no eixo X
    plt.tight_layout()
    output_filepath_lemmas = os.path.join(output_dir, 'grafico_lemmas_frequentes_horizontal.png') # Novo nome
    plt.savefig(output_filepath_lemmas, dpi=300, bbox_inches='tight')
    print(f"\nGráfico de lemas frequentes (horizontal) salvo em: {output_filepath_lemmas}")
    plt.show()
    print("Geração do gráfico de lemas concluída.")


print("\nAnálise concluída. Verifique os gráficos na pasta 'graficos_analise'.")