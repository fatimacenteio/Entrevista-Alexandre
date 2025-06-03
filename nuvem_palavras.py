import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. O Corpus da Entrevista (a sua transcrição completa)
transcricao = """
Boa tarde. Neste momento, inicio a minha entrevista com o meu colega Alexandre.
Pergunta: Que palavras usarias para descrever a tua infância?
Resposta: Divertida, única, empolgante, inesquecível. Acho que "inesquecível" é a palavra-chave.
Pergunta: Onde cresceste e como era o sítio onde vivias?
Resposta: Cresci na minha freguesia, onde ainda vivo atualmente, em Nino. Foi lá que fiz muitos dos meus amigos de infância, de juventude — amigos que ainda fazem parte da minha vida. Todas as minhas memórias de infância, pré-adolescência e adolescência são de lá.
Pergunta: Tens alguma memória especial da tua infância que te tenha marcado?
Resposta: Tenho várias. Por exemplo, quando íamos à catequese, jogávamos sempre depois — esconde-esconde, jogo da macaca, lencinho. São memórias que levarei comigo. Jogar à bola com o McLaren, as primeiras saídas só com os amigos, festas de aniversário, Carnaval, Natal, Páscoa — tudo isso me marcou muito.
Pergunta: Qual era a tua brincadeira preferida quando eras criança?
Resposta: Adorava jogar à apanhada e aos "pontos-pontos". Sempre que nos juntávamos, fosse durante a semana ou nas férias, íamos para o parque e jogávamos. Era uma rotina.
Pergunta: Havia algum brinquedo ou objeto que fosse especial para ti?
Resposta: Não havia um brinquedo específico, mas guardo com carinho algumas peças de vestuário do tempo em que jogava futebol. T-shirts com o meu nome, chuteiras, casacos... evocam boas memórias da infância.
Pergunta: Como foi a tua experiência na escola primária? Tinhas alguma disciplina preferida?
Resposta: Sempre gostei muito de inglês. Mesmo quando não entendia os filmes, sentia vontade de compreender. Também gostava de história e, claro, de futebol.
Pergunta: Lembras-te do teu primeiro amigo de infância? Ainda manténs contacto com ele?
Resposta: Sim, claro. É filho da melhor amiga da minha mãe. Crescemos juntos. Dormíamos em casa um do outro, passámos fins de semana e férias juntos. Temos uma amizade sólida e mantemos contacto até hoje.
Pergunta: Qual era a tua rotina diária durante a infância?
Resposta: Acordava cedo, sempre bem disposto. Passava os dias entre casa, o infantário e, quando a minha avó estava em Portugal, ficava com ela. A rotina era simples, mas marcante.
Pergunta: Tinhas alguma tradição familiar especial de que te lembres com carinho?
Resposta: Hoje em dia temos uma tradição: aos domingos, o meu avô e o meu tio materno almoçam lá em casa. Desde pequeno que comemoramos o Natal e a Páscoa sempre juntos, na nossa casa. Isso sempre foi uma constante.
Pergunta: Havia alguma comida que adoravas ou detestavas quando eras criança?
Resposta: Detestava pepino. Quando íamos ao McDonald's, pedia logo à minha mãe para tirar o pepino. Hoje é um dos meus vegetais favoritos. Ainda não consigo comer gordura da carne, mas como quase tudo hoje em dia.
Pergunta: O que desejavas ser quando eras criança?
Resposta: Queria ser médico, para curar os diabetes da minha avó paterna. Mais tarde percebi que não tem cura, mas essa era a minha motivação. Também quis ser polícia, influenciado por séries americanas.
Pergunta: Se pudesses reviver um dia da tua infância, qual escolherias e porquê?
Resposta: Um dia de Natal, com a minha avó materna e o meu avô paterno ainda vivos. Estarmos todos juntos na mesma mesa seria o dia perfeito para reviver. Não hesitaria.
É tudo.
"""

# 2. Carregar o modelo SpaCy para português
# Certifique-se de que já fez: python -m spacy download pt_core_news_md
try:
    nlp = spacy.load("pt_core_news_md")
except OSError:
    print("Modelo 'pt_core_news_md' não encontrado. Tentando fazer o download...")
    from spacy.cli import download
    download("pt_core_news_md")
    nlp = spacy.load("pt_core_news_md")

# 3. Pré-processamento do Texto
# Remover "Pergunta:" e "Resposta:" e unificar o texto
texto_limpo = transcricao.replace("Pergunta:", "").replace("Resposta:", "").replace("Boa tarde. Neste momento, inicio a minha entrevista com o meu colega Alexandre.", "").replace("É tudo.", "").strip()

doc = nlp(texto_limpo)

# Extrair lemas, remover stopwords e pontuação, e converter para minúsculas
palavras_processadas = []
for token in doc:
    # Apenas palavras (não pontuação, espaços, etc.) e que não sejam stopwords
    if token.is_alpha and not token.is_stop:
        palavras_processadas.append(token.lemma_.lower()) # Lematizar e converter para minúsculas

# Juntar as palavras processadas em uma única string para a nuvem de palavras
texto_para_wordcloud = " ".join(palavras_processadas)

# 4. Gerar a Nuvem de Palavras
# Configurações da nuvem de palavras (pode ajustar as cores, tamanho, etc.)
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis', # Ou 'plasma', 'cividis', 'magma', 'inferno'
    max_words=100,
    contour_color='steelblue', # Borda das palavras
    contour_width=1,
    collocations=False # Evita que "amigos de" apareça junto se "de" for stopword
).generate(texto_para_wordcloud)

# 5. Visualizar a Nuvem de Palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off') # Remove os eixos
plt.title('Nuvem de Palavras das Memórias de Infância de Alexandre')
plt.show()

# 6. Salvar a Nuvem de Palavras (opcional)
wordcloud.to_file("nuvem_palavras_alexandre.png")
print("Nuvem de palavras salva como nuvem_palavras_alexandre.png")