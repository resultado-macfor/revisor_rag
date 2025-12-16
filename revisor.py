import openai
import os
import json
import hashlib
from typing import List, Dict, Optional
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from revisor import reescrever_revisor, get_embedding, ajuste_incremental
except ImportError as e:
    st.error(f"❌ ERRO DE IMPORTAÇÃO: {e}. Verifique se todos os arquivos estão no diretório correto.")
    st.stop()


# 🚨 IMPORTAÇÃO DOS MÓDULOS DE LÓGICA
try:
    from classificacao import classificar_texto 
    print("✅ Módulo 'classificacao' importado.")
    from conexao_banco import AstraDBClient, astra_client
    print("✅ Módulo 'conexao_banco' importado.")
except ImportError as e:
    print(f"❌ ERRO: Verifique se os arquivos classificacao.py e conexao_banco.py estão no diretório. Erro: {e}")
    # Abortar se as dependências não puderem ser carregadas
    exit()




OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define a chave de ambiente para o cliente OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if not OPENAI_API_KEY:
    print("❌ ATENÇÃO: OPENAI_API_KEY não está definida.")



if 'secrets' in dir(st) and st.secrets:
    try:
        # Carregar todas as secrets
        for key in ['OPENAI_API_KEY', 'GEMINI_API_KEY', 'ASTRA_DB_APPLICATION_TOKEN', 
                    'ASTRA_DB_API_ENDPOINT', 'ASTRA_DB_NAMESPACE']:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception as e:
        st.error(f"Erro ao carregar secrets: {e}")
else:
    st.warning("⚠️ Secrets não encontrados. Usando variáveis de ambiente existentes.")
    

# -----------------------------------------------------------
# II. CLASSE LLMClient (Para gerar a correção)
# -----------------------------------------------------------

class LLMClient:
    """Classe wrapper para o cliente de Chat Completion da OpenAI, simulando 'generate_content'."""
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        # Inicializa o cliente OpenAI
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        print(f"✅ LLMClient inicializado com modelo: {self.model}")

    def generate_content(self, prompt: str) -> str:
        """Método que simula a interface generate_content."""
        print("\n--- Chamando OpenAI Chat Completion ---")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Você é um agente de revisão técnica altamente preciso."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            print(f"❌ ERRO NA GERAÇÃO DO LLM (API Error): {e}")
            return f"ERRO NA GERAÇÃO DO LLM (API Error): {str(e)}"
        except Exception as e:
            print(f"❌ ERRO NA GERAÇÃO DO LLM (Geral): {e}")
            return f"ERRO NA GERAÇÃO DO LLM (Geral): {str(e)}"

# Inicializa o cliente
modelo_texto = LLMClient(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# III. FUNÇÃO get_embedding (Para a busca vetorial)
# -----------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """Obtém embedding do texto usando OpenAI com diagnóstico (adaptado do seu doc)."""
    print("\n--- Chamando OpenAI Embedding ---")
    try:
        # Usa o cliente já inicializado para embeddings
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # --- DIAGNÓSTICO ---
        print(f"✅ Embedding Gerado. Dimensões: {len(embedding)}. Primeiro valor: {embedding[0]:.6f}")
        # --- FIM DIAGNÓSTICO ---

        return embedding
    except Exception as e:
        print(f"❌ ERRO na API OpenAI para Embedding: {str(e)}. Verifique se a chave está ativa.")
        # Seu fallback de hash foi removido, pois ele falha na busca RAG e queremos testar a conexão real.
        return []



def reescrever_revisor(content: str, colecao_override: Optional[str] = None) -> str:
    """
    Função principal que executa o pipeline RAG completo.
    Atua como um Revisor Técnico, corrigindo imprecisões e enriquecendo o texto.
    Aceita colecao_override para sobrepor a classificação do Gemini.
    """
    
    colecao = None
    
    if colecao_override and colecao_override != "Automática (Classificação Gemini)":
        # 1a. Usa a coleção fornecida pelo usuário
        colecao = colecao_override
        print(f"\n--- 1. COLEÇÃO DEFINIDA PELO USUÁRIO: {colecao} ---")
    else:
        # 1b. Executa a classificação normal do Gemini
        print("\n--- 1. CLASSIFICAÇÃO AUTOMÁTICA (Gemini) ---")
        colecao = classificar_texto(content)
        print(f"Coleção Identificada: {colecao}")
    
    if colecao in ["ERRO", "CLASSIFICAÇÃO NÃO RECONHECIDA:", None]:
        # Retorna a mensagem de erro como string, conforme solicitado.
        return f"Erro na classificação/seleção da coleção. Classificação falhou com: {colecao if colecao else 'ERRO'}. Não foi possível iniciar a busca RAG."

    # 2. EMBEDDING E BUSCA
    embedding = get_embedding(content[:800])
    
    if not embedding or len(embedding) < 1536:
        return "Erro fatal na geração do Embedding. Verifique sua chave OpenAI ativa. Não foi possível buscar no Astra DB."
        
    relevant_docs = astra_client.vector_search(colecao, embedding, limit=10)
    print(f"2. Busca Vetorial concluída na coleção '{colecao}'. Documentos retornados: {len(relevant_docs)}")
    
    # 3. CONSTRÓI CONTEXTO RAG
    rag_context = ""
    if relevant_docs:
        rag_context = "### REFERENCIAL TEÓRICO BUSCADO (RAG) ###\n"
        for i, doc in enumerate(relevant_docs, 1):
            doc_content = str(doc)
            doc_clean = doc_content.replace('{', '').replace('}', '').replace("'", "").replace('"', '')
            rag_context += f"--- Fonte {i} ---\n{doc_clean[:500]}...\n"
    else:
        rag_context = "Referencial teórico não retornou resultados específicos relevantes."
    
    # 4. PROMPT DE GERAÇÃO AUMENTADA (Mantendo o prompt anterior, mas removendo a 'instrucao_incremental')
    final_prompt = f"""
    Você é um **Revisor Técnico Sênior** com foco na área agrícola, rigoroso, preciso e com a missão de garantir a **veracidade científica absoluta** do texto de entrada.
    Confira se os valores estão idênticos ao banco de dados.

    Seu objetivo é:
    1. **CORRIGIR** automaticamente qualquer imprecisão, erro técnico ou erro científico no texto original.
    2. **ENRICHECER** o texto original, substituindo termos vagos por **terminologia técnica precisa** (ex: troque 'veneno' por 'defensivo agrícola' ou 'fitossanitário').
    3. **ACRESCENTAR** dados concretos, números e informações específicas, *apenas* quando o **REFERENCIAL TEÓRICO** fornecido for relevante para enriquecer ou corrigir o tópico do texto original.
    4. **MANTER** a estrutura e o tamanho do texto original (máximo delta de 5%).
    5. **PROIBIDO** adicionar informações que tangenciem ou desviem do tema central do texto original.

    ---
    ### TEXTO ORIGINAL A SER REVISADO ###
    {content}
    
    ---
    {rag_context}
    ---

    ## ESTRUTURA DE RETORNO OBRIGATÓRIA:

    Retorne o **TEXTO COMPLETAMENTE REVISADO E CORRIGIDO** primeiro.
    
    Após, coloque quais dados foram buscados no banco de dados para essa correção.

    Em seguida, adicione uma subseção chamada "🛠️ Ajustes Técnicos e Correções" listando de forma concisa cada alteração significativa feita (correção ou enriquecimento) e qual fonte foi usada.
    """

    # 5. Geração Final do LLM
    response_text = modelo_texto.generate_content(final_prompt)
        
    return response_text





# -----------------------------------------------------------
# V. FUNÇÃO ajuste_incremental (Para ajustes pós-revisão)
# -----------------------------------------------------------
# -----------------------------------------------------------
# V. FUNÇÃO ajuste_incremental (Para ajustes pós-revisão)
# -----------------------------------------------------------

def ajuste_incremental(texto_revisado: str, instrucao_incremental: str) -> str:
    """
    Aplica uma instrução incremental ao texto já revisado (saída do reescrever_revisor).
    Mantém o formato e adiciona as mudanças solicitadas.
    """
    if not instrucao_incremental:
        return texto_revisado # Retorna o texto original se não houver instrução

    print("\n--- INICIANDO AJUSTE INCREMENTAL ---")
    
    # 1. TENTA ISOLAR APENAS O TEXTO PRINCIPAL DA SAÍDA RAG
    # Isso é crucial para evitar que o LLM inclua as seções de metadados (Ajustes Técnicos) na resposta
    partes = texto_revisado.split("🛠️ Ajustes Técnicos e Correções")
    texto_principal_rag = partes[0].strip()
    
    # PROMPT DE AJUSTE INCREMENTAL REFINADO
    final_prompt = f"""
    Você é um **Editor Sênior** com a única missão de aplicar uma mudança incremental de forma fluida.
    
    Seu objetivo principal é editar o TEXTO PRINCIPAL A SER AJUSTADO:
    1. **APENAS** edite o texto para incorporar as informações da INSTRUÇÃO INCREMENTAL de forma natural, **mantendo o tom técnico**.
    2. Não é para mencionar a instrução incremental na saída.
    3. **PROIBIDO** manter ou incluir as seções de metadados ("🛠️ Ajustes Técnicos e Correções", "Dados Buscados", etc.) na sua resposta.

    ---
    ### TEXTO PRINCIPAL A SER AJUSTADO ###
    {texto_principal_rag}
    
    ---
    ### INSTRUÇÃO INCREMENTAL A SER ACRESCENTADA ###
    {instrucao_incremental}

    ---
    
    Retorne **SOMENTE O TEXTO FINAL RESULTANTE**, completamente editado e pronto.
    """

    try:
        # Usa o cliente LLM para gerar o conteúdo
        response_text = modelo_texto.generate_content(final_prompt)
        print("✅ Ajuste Incremental concluído.")
        return response_text
    except Exception as e:
        print(f"❌ ERRO na Geração do Ajuste Incremental: {str(e)}")
        return texto_revisado # Fallback para o texto original se falhar
# -----------------------------------------------------------
# V. TESTE PRINCIPAL (main) - EXATAMENTE COMO SOLICITADO
# -----------------------------------------------------------

# -----------------------------------------------------------
# VI. TESTE PRINCIPAL (main)
# -----------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("--- TESTE FINAL DO MÓDULO REVISOR.PY ---")
    print("=" * 70)
    
    # Etapa 1: Revisão Principal (RAG)
    texto_base = input("Insira o TEXTO BASE para revisão: ")
    override = input("FORÇAR COLEÇÃO? (Ex: 'Defensivos', ou deixe vazio para Classificação Automática): ")
    
    colecao_usada = override if override else None
    
    if not texto_base:
        print("Entrada base vazia. Abortando.")
    else:
        # 1. Executa a Revisão RAG Principal
        resultado_rag = reescrever_revisor(texto_base, colecao_override=colecao_usada)
        
        print("\n" + "=" * 70)
        print("✅ REVISÃO RAG FINALIZADA")
        print("=" * 70)
        print("\n### RESULTADO RAG COMPLETO ###")
        print(resultado_rag)
        print("=" * 70)

        # Etapa 2: Ajuste Incremental
        print("\n" + "#" * 30 + " SEGUNDA ETAPA " + "#" * 30)
        instrucao = input("Insira a INSTRUÇÃO INCREMENTAL (Deixe vazio para finalizar): ")

        if instrucao:
            # 2. Executa o Ajuste Incremental no resultado do RAG
            resultado_final = ajuste_incremental(resultado_rag, instrucao)
            
            print("\n" + "=" * 70)
            print("✨ AJUSTE INCREMENTAL CONCLUÍDO")
            print("=" * 70)
            print("\n### RESULTADO FINAL APÓS AJUSTE INCREMENTAL ###")
            print(resultado_final)
        else:
            resultado_final = resultado_rag
            print("Nenhuma instrução incremental fornecida. O resultado final é o resultado RAG.")
            
        print("=" * 70)
