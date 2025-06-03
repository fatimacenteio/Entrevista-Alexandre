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
# 1. Remover duplicações: Divide o texto por "Fim da entrevista." e pega a primeira parte.
# Isso evita que o texto seja processado múltiplas vezes se tiver sido copiado/colado duas vezes.
partes = texto_entrevista_original.split("Fim da entrevista.")
texto_base = partes[0].strip()

# 2. Remover "Pergunta:" e "Resposta:" usando regex (ignora maiúsculas/minúsculas).
texto_limpo_regex = re.sub(r'(?i)Pergunta:\s*|Resposta:\s*', '', texto_base).strip()

# 3. Remover linhas vazias extras que podem ter resultado da limpeza, e garante que o texto não fica vazio.
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
# Essas palavras, mesmo que o SpaCy as classifique como MISC, não são "entidades nomeadas" para nosso propósito.
# Inclui verbos, adjetivos, advérbios, pronomes, substantivos comuns, etc.
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
    # --- ADIÇÕES IMPORTANTES PARA EXCLUIR OS VERBOS NO GRÁFICO ---
    "guardo", "lembras-te", "manténs", "acordava"
}
# Converte a lista de exclusão para um set para buscas mais rápidas.
excluded_common_words_set = set(common_words_to_exclude)

# --- Extração e Análise de Entidades Nomeadas ---
print("\n--- Analisando Entidades Nomeadas ---")
entidades_para_contagem = [] # Armazena as entidades em minúsculas para a contagem unificada.
entidades_para_exibicao_map = {} # Mapeia a versão em minúsculas para a versão capitalizada correta para exibição no gráfico.

# Definir a lista de preposições a limpar no início de uma entidade
preposicoes_a_limpar_inicio = {"em", "de", "da", "do", "das", "dos", "para", "a", "à", "ao", "aos", "às", "com", "por"}

for ent in doc.ents:
    # Filtra por tipos de entidades relevantes: Pessoa (PER), Local (LOC), Organização (ORG), Miscelânea (MISC).
    if ent.label_ in ["PER", "LOC", "ORG", "MISC"]:
        ent_text_stripped = ent.text.strip()
        
        # --- LÓGICA DE CORREÇÃO: Tenta remover preposições iniciais para entidades ---
        for prep in preposicoes_a_limpar_inicio:
            # Verifica se a entidade começa com a preposição seguida de um espaço (case-insensitive)
            if ent_text_stripped.lower().startswith(prep + " "):
                # Remove a preposição e o espaço, e remove espaços extra que possam surgir
                ent_text_stripped = ent_text_stripped[len(prep) + 1:].strip()
                # Opcional: Se a entidade restante for muito curta ou vazia, pode-se decidir ignorá-la.
                if not ent_text_stripped:
                    break # Se ficou vazio, sai do loop de preposições e a entidade será tratada no filtro de exclusão abaixo
                break # Sai do loop de preposições após a primeira correspondência
        # --- FIM DA LÓGICA DE CORREÇÃO ---

        ent_text_lower = ent_text_stripped.lower()

        # CRUCIAL: Se a entidade (já limpa de preposição) estiver na nossa lista de exclusão de palavras comuns, IGNORA-A COMPLETAMENTE.
        if ent_text_lower in excluded_common_words_set:
            continue # Pula para a próxima entidade no loop.

        # Adiciona a versão em minúsculas para a lista de contagem.
        entidades_para_contagem.append(ent_text_lower)

        # Lógica para tentar capturar a capitalização mais "correta" que o SpaCy identificou.
        # Preferimos a versão original se capitalizada, ou a primeira encontrada.
        if ent_text_lower not in entidades_para_exibicao_map:
            entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped
        # Se a entidade em minúsculas já existe no mapa, mas a nova versão está capitalizada (e a armazenada não), atualiza.
        elif ent_text_stripped[0].isupper() and not entidades_para_exibicao_map[ent_text_lower][0].isupper():
            entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped
        # Para termos multi-palavras que já vêm capitalizados corretamente do SpaCy (ex: "McDonald's").
        elif ' ' in ent_text_stripped and (ent_text_stripped.istitle() or ent_text_stripped[0].isupper()):
            # Se for multi-palavra e já estiver em Title Case ou começar com maiúscula, preferir.
             entidades_para_exibicao_map[ent_text_lower] = ent_text_stripped


# --- Regras de Capitalização Forçada para Nomes Próprios Específicos e Termos Chave ---
# Esta parte garante que os termos importantes apareçam *exatamente* como queremos no gráfico,
# sobrepondo qualquer capitalização automática do SpaCy ou da lógica anterior.
# Mantenha esta lista apenas para as entidades que VOCÊ QUER que apareçam capitalizadas de uma forma específica.
termos_a_capitalizar_corretamente = {
    "alexandre": "Alexandre",
    "mclaren": "McLaren",
    "nino": "Nino",
    "natal": "Natal",
    "páscoa": "Páscoa",
    "carnaval": "Carnaval",
    "inglês": "Inglês",
    "mcdonald's": "McDonald's",
    "catequese": "Catequese", # Atividade/Instituição específica
    "história": "História", # Embora excluído da contagem, se fosse para aparecer por outra razão, teria esta capitalização.
    "portugal": "Portugal" # Adicionado para garantir capitalização correta caso seja extraído sem "em"
}

# Aplica as regras de capitalização forçada ao mapa de exibição.
for key_lower, correct_capitalization in termos_a_capitalizar_corretamente.items():
    if key_lower in entidades_para_exibicao_map:
        entidades_para_exibicao_map[key_lower] = correct_capitalization


# Contar a frequência das entidades (usando as versões em minúsculas para unificar a contagem).
frequencia_entidades_lower = Counter(entidades_para_contagem)

# Criar um DataFrame para o gráfico, usando as versões capitalizadas corretas para exibição.
df_entidades_data = []
# Pega as 10 entidades mais comuns do contador de minúsculas.
for ent_lower, freq in frequencia_entidades_lower.most_common(10):
    # Obtém o nome para exibição do dicionário de exibição (`entidades_para_exibicao_map`).
    # Se não encontrar (o que é raro após toda a lógica), capitaliza a primeira letra como fallback.
    display_name = entidades_para_exibicao_map.get(ent_lower, ent_lower.capitalize())

    # Adiciona a entidade e sua frequência à lista de dados para o DataFrame.
    df_entidades_data.append([display_name, freq])

df_entidades = pd.DataFrame(df_entidades_data, columns=['Entidade', 'Frequência'])


# Verifica se há entidades para plotar antes de tentar gerar o gráfico.
if df_entidades.empty:
    print("Nenhuma entidade nomeada relevante encontrada no texto para gerar o gráfico após a filtragem.")
    exit()

print("Tabela de Entidades Nomeadas e Frequência:")
print(df_entidades) # Imprime a tabela de entidades e frequências no console.

# --- Geração do Gráfico de Entidades Nomeadas ---
plt.figure(figsize=(12, 8)) # Define o tamanho da figura para melhor visualização.

# Usa um colormap (viridis) para dar cores diferentes às barras, baseadas na frequência.
colors = plt.cm.viridis(df_entidades['Frequência'] / df_entidades['Frequência'].max())

# Cria um gráfico de barras horizontais.
plt.barh(df_entidades['Entidade'], df_entidades['Frequência'], color=colors)

plt.xlabel('Frequência', fontsize=12) # Rótulo do eixo X (Frequência).
plt.ylabel('Entidade', fontsize=12) # Rótulo do eixo Y (Entidade).
plt.title('Extração de Entidades Nomeadas', fontsize=14) # <-- Título do gráfico alterado aqui.

plt.gca().invert_yaxis() # Inverte o eixo Y para que a entidade mais frequente fique no topo.
plt.grid(axis='x', linestyle='--', alpha=0.7) # Adiciona linhas de grade verticais para facilitar a leitura.
plt.tight_layout() # Ajusta o layout para evitar sobreposição de elementos.

# Salvar o gráfico como um ficheiro PNG de alta resolução.
output_filepath = os.path.join(output_dir, 'grafico_entidades_nomeadas_final_corrigido.png')
plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
print(f"\nGráfico de entidades nomeadas salvo em: {output_filepath}")

plt.show() # Exibe o gráfico.

print("\nGeração do gráfico de entidades concluída.")