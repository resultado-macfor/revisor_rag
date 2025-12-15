import streamlit as st
import os
# 🚨 IMPORTAÇÃO ATUALIZADA: Agora importa reescrever_revisor E ajuste_incremental
from revisor import reescrever_revisor, get_embedding, ajuste_incremental 

# --- Configurações da Página ---
st.set_page_config(
    page_title="Corretor de Texto ",
    layout="wide"
)

# --- Título e Status Inicial ---
st.title("🛠️ Corretor de Texto ")
# 🚨 Descrição do fluxo atualizada para refletir as duas etapas
st.markdown("**Fluxo de Duas Etapas:** 1. Revisão RAG (Classificação/Busca) ➡️ 2. Ajuste Incremental (Se houver)")
st.markdown("---")

# --- Verificação de Status da Chave OpenAI ---
# Nota: A função get_embedding não é ideal para check, mas mantida para compatibilidade com o revisor.py
if not get_embedding("teste"):
    st.error("❌ ERRO CRÍTICO: Chave OpenAI INATIVA. A busca RAG falhará. Por favor, corrija a chave no 'revisor.py'.")
else:
    st.success("✅ Conexão OpenAI OK. Pronto para rodar o RAG.")
st.markdown("---")

# --- Variáveis de Estado (Simples) ---
if 'saida_final' not in st.session_state:
    st.session_state.saida_final = ""
if 'ajustes_tecnicos' not in st.session_state:
    st.session_state.ajustes_tecnicos = "Nenhum ajuste técnico realizado."
if 'colecao_usada' not in st.session_state:
    st.session_state.colecao_usada = "N/A"

# --- FUNÇÃO AUXILIAR PARA PARSEAR A SAÍDA DO RAG ---
# Como reescrever_revisor retorna uma string única, precisamos extrair o texto final e os ajustes.
def parse_rag_output(full_response: str, colecao: str) -> dict:
    if "Erro na classificação" in full_response or "Erro fatal na geração do Embedding" in full_response:
        return {
            "texto_final": full_response,
            "ajustes_tecnicos": "Falha na Etapa RAG.",
            "colecao_usada": colecao
        }

    # Tenta separar o texto principal dos ajustes técnicos
    partes = full_response.split("🛠️ Ajustes Técnicos e Correções")
    texto_final = partes[0].strip() if partes else full_response
    ajustes_tecnicos = partes[1].strip() if len(partes) > 1 else "Não foi possível extrair a seção de Ajustes Técnicos."
        
    return {
        "texto_final": texto_final,
        "ajustes_tecnicos": ajustes_tecnicos,
        "colecao_usada": colecao
    }


# --- 1. Seção de Entradas ---
st.header("Entradas do Usuário")

col1, col2 = st.columns(2)

with col1:
    texto_base = st.text_area(
        label="Texto Base para Revisão:",
        height=250, 
        placeholder="Insira o texto original aqui.",
    )

with col2:
    # Seletor Opcional de Coleção
    colecoes_disponiveis = [
        "Automática (Classificação Gemini)", # Opção padrão
        "PRODUTO",
        "CULTURA",
        "OUTROS"
    ]
    colecao_selecionada = st.selectbox(
        label="Escolha Opcional da Coleção Astra DB:",
        options=colecoes_disponiveis,
        index=0, # Inicia na opção automática
        help="Selecione uma coleção específica para busca RAG. Se 'Automática' for escolhida, a classificação Gemini será usada."
    )
    
    instrucao_incremental = st.text_area(
        label="Instrução Adicional/Incremental (Opcional):",
        height=150,
        placeholder="Ex: 'Mude o tom para formal' ou 'Aumente o segundo parágrafo em 30 palavras'."
    )
    
# --- Lógica de Execução ---

st.markdown("---")

if st.button("Aplicar Correção", type="primary"):
    
    if not texto_base:
        st.warning("Por favor, insira um Texto Base para revisão.")
    else:
        # Inicializa o resultado final com o texto base em caso de falha
        final_text = texto_base

        # ----------------------------------------------------
        # 🟢 PASSO 1: REVISÃO RAG (reescrever_revisor)
        # ----------------------------------------------------
        with st.spinner(f"1/2 Processando RAG na coleção: {colecao_selecionada}..."):
            # CHAMA A FUNÇÃO CENTRAL DO RAG
            rag_output_str = reescrever_revisor(texto_base, colecao_override=colecao_selecionada)
            
            # PARSEA A SAÍDA PARA SEPARAR O TEXTO FINAL E OS AJUSTES
            resultado_rag_parse = parse_rag_output(rag_output_str, colecao_selecionada)
            
            st.session_state.ajustes_tecnicos = resultado_rag_parse["ajustes_tecnicos"]
            st.session_state.colecao_usada = resultado_rag_parse["colecao_usada"]
            final_text = resultado_rag_parse["texto_final"]
            
            if "Erro" in final_text:
                st.error(f"❌ Erro na Etapa RAG: {final_text}")
            else:
                st.success(f"✅ Etapa 1 (RAG) Concluída. Coleção utilizada: {st.session_state.colecao_usada}")

        # ----------------------------------------------------
        # 🟠 PASSO 2: AJUSTE INCREMENTAL (ajuste_incremental)
        # ----------------------------------------------------
        if instrucao_incremental and "Erro" not in final_text:
            with st.spinner("2/2 Aplicando Ajuste Incremental..."):
                final_text = ajuste_incremental(final_text, instrucao_incremental)
            
            st.success("✨ Ajuste Incremental Aplicado.")
            st.session_state.ajustes_tecnicos += "\n\n--- AJUSTE INCREMENTAL ---\nInstrução Adicional Aplicada."
        elif instrucao_incremental and "Erro" in final_text:
             st.warning("Instrução incremental ignorada devido a um erro na etapa RAG.")


        # ----------------------------------------------------
        # 🏁 ATUALIZAÇÃO FINAL
        # ----------------------------------------------------
        st.session_state.saida_final = final_text

st.markdown("---")

# --- 2. Seção de Saída (Resultado Final) ---
st.header("Resultado Final")

# O resultado principal (texto limpo + dados buscados)
st.text_area(
    label="Texto Corrigido/Final (Resultado do RAG + Ajuste Incremental, se houver):",
    value=st.session_state.saida_final,
    height=450,
    disabled=True 
)

# A seção de ajustes técnicos e fontes (detalhes do RAG)
st.subheader("🛠️ Detalhes da Revisão")
st.code(
    f"Coleção RAG Utilizada: {st.session_state.colecao_usada}\n\n" + st.session_state.ajustes_tecnicos,
    language='markdown'
)
