import streamlit as st
import requests
import time
import os
from PIL import Image
from io import BytesIO

# Config API endpoints
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Recomendação Visual & Busca Multimodal",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Recomendação Visual & Busca Multimodal")
st.markdown("""
Esta demonstração interativa apresenta as capacidades de um sistema de busca visual e recomendação inteligente para e-commerce.
Utilizando o modelo multimodal **OpenAI CLIP** (`clip-vit-base-patch32`), o sistema projeta imagens e descrições textuais no mesmo espaço vetorial de 512 dimensões.
""")

# Setup navigation tabs
tab_search, tab_catalog, tab_admin = st.tabs(["🔍 Buscar Produtos", "📦 Catálogo", "⚙️ Admin & Setup"])

# Helper functions
def search_by_image(file_bytes, top_k, category):
    files = {"file": ("image.jpg", file_bytes, "image/jpeg")}
    params = {"top_k": top_k}
    if category and category != "Todas":
        params["categoria"] = category
    try:
        start_time = time.time()
        res = requests.post(f"{API_URL}/api/search/image", files=files, params=params)
        latency = (time.time() - start_time) * 1000
        if res.status_code == 200:
            return res.json(), latency
        else:
            st.error(f"Erro na busca por imagem: {res.text}")
    except Exception as e:
        st.error(f"Não foi possível conectar à API: {e}")
    return [], 0

def search_by_text(query, top_k, category):
    params = {"query": query, "top_k": top_k}
    if category and category != "Todas":
        params["categoria"] = category
    try:
        start_time = time.time()
        res = requests.post(f"{API_URL}/api/search/text", params=params)
        latency = (time.time() - start_time) * 1000
        if res.status_code == 200:
            return res.json(), latency
        else:
            st.error(f"Erro na busca por texto: {res.text}")
    except Exception as e:
        st.error(f"Não foi possível conectar à API: {e}")
    return [], 0

def get_similar_products(product_id, top_k, category):
    params = {"top_k": top_k}
    if category and category != "Todas":
        params["categoria"] = category
    try:
        start_time = time.time()
        res = requests.get(f"{API_URL}/api/products/similar/{product_id}", params=params)
        latency = (time.time() - start_time) * 1000
        if res.status_code == 200:
            return res.json(), latency
        else:
            st.error(f"Erro ao buscar recomendações: {res.text}")
    except Exception as e:
        st.error(f"Não foi possível conectar à API: {e}")
    return [], 0

def list_products():
    try:
        res = requests.get(f"{API_URL}/api/products")
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        st.warning(f"Não foi possível listar produtos: {e}")
    return []

# Sidebar configurations
st.sidebar.header("🔧 Configurações")
top_k = st.sidebar.slider("Quantidade de resultados (Top-K)", min_value=1, max_value=20, value=6)

# Load existing products to populate categories list
catalog = list_products()
categories = ["Todas"] + list(set([p["categoria"] for p in catalog])) if catalog else ["Todas"]
selected_category = st.sidebar.selectbox("Filtrar por Categoria", categories)

# --- TAB 1: SEARCH ---
with tab_search:
    st.header("🔍 Multi-Search Engine")
    search_mode = st.radio("Selecione o modo de busca:", ["Buscar por Foto (Visual Search)", "Busca Híbrida / Zero-Shot (Texto)", "Similares por ID (Recomendação)"])

    results = []
    latency = 0

    if search_mode == "Buscar por Foto (Visual Search)":
        st.subheader("📸 Faça o upload de uma imagem do produto")
        uploaded_file = st.file_uploader("Arraste e solte ou selecione uma foto de um item (ex: cadeira, roupa, calçado)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            col_img, col_act = st.columns([1, 2])
            with col_img:
                st.image(file_bytes, caption="Sua Imagem", width=250)
            with col_act:
                if st.button("🚀 Encontrar Semelhantes"):
                    with st.spinner("Analisando imagem e calculando proximidade semântica..."):
                        results, latency = search_by_image(file_bytes, top_k, selected_category)

    elif search_mode == "Busca Híbrida / Zero-Shot (Texto)":
        st.subheader("📝 Busque por texto livre (Zero-Shot)")
        query = st.text_input("Digite o que procura (ex: 'vestido vermelho para festa', 'cadeira de madeira minimalista', 'tênis de corrida azul'):")

        if st.button("🔍 Buscar"):
            if query.strip():
                with st.spinner("Traduzindo consulta textual em vetor CLIP..."):
                    results, latency = search_by_text(query, top_k, selected_category)
            else:
                st.warning("Por favor, insira um termo de busca.")

    elif search_mode == "Similares por ID (Recomendação)":
        st.subheader("🔗 Recomendações na Página do Produto")
        if catalog:
            product_options = {f"{p['id']} - {p['nome_produto']} ({p['categoria']})": p['id'] for p in catalog}
            selected_prod_label = st.selectbox("Selecione um produto de referência do catálogo:", list(product_options.keys()))
            selected_id = product_options[selected_prod_label]

            # Show current product
            curr_prod = next((p for p in catalog if p['id'] == selected_id), None)
            if curr_prod:
                st.info(f"Visualizando recomendações para o item: **{curr_prod['nome_produto']}** | Preço: R$ {curr_prod['preco']:.2f}")

            if st.button("🎁 Gerar Recomendações Semelhantes"):
                with st.spinner("Buscando vetores mais próximos do produto selecionado..."):
                    results, latency = get_similar_products(selected_id, top_k, selected_category)
        else:
            st.warning("O catálogo está vazio. Cadastre alguns produtos na aba 'Admin & Setup' ou re-inicialize o banco.")

    # Render results grid
    if results:
        st.success(f"Busca finalizada com sucesso! Latência do Modelo & Busca: **{latency:.2f} ms**")
        st.markdown("### 🌟 Produtos Recomendados")

        # Display in columns
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            row_items = results[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx, item in enumerate(row_items):
                with cols[idx]:
                    st.image(item["url_imagem"], use_container_width=True)
                    st.write(f"**{item['nome_produto']}**")
                    st.write(f"🏷️ Categoria: *{item['categoria']}*")
                    st.write(f"💵 Preço: R$ {item['preco']:.2f}")
                    # Convert to percentage
                    sim_score = item["score"] * 100
                    st.metric("Similaridade Visual", f"{sim_score:.1f}%")
                    st.markdown("---")

# --- TAB 2: CATALOG ---
with tab_catalog:
    st.header("📦 Catálogo de Itens Cadastrados")
    if catalog:
        st.write(f"Exibindo {len(catalog)} produtos no banco de dados.")

        cols_per_row = 4
        for i in range(0, len(catalog), cols_per_row):
            row_items = catalog[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx, item in enumerate(row_items):
                with cols[idx]:
                    st.image(item["url_imagem"], use_container_width=True)
                    st.write(f"**{item['nome_produto']}**")
                    st.write(f"🆔 ID: `{item['id']}`")
                    st.write(f"🏷️ {item['categoria']}")
                    st.write(f"💵 R$ {item['preco']:.2f}")
                    st.markdown("---")
    else:
        st.warning("O catálogo de produtos está vazio no momento.")

# --- TAB 3: ADMIN & SETUP ---
with tab_admin:
    st.header("⚙️ Administração & Configurações de Ingestão")

    col_setup, col_ingest = st.columns(2)

    with col_setup:
        st.subheader("🛠️ Inicialização do Banco")
        st.write("Clique abaixo para limpar e recriar o índice HNSW no OpenSearch (ou base local).")
        if st.button("🗑️ Resetar / Inicializar Banco de Dados"):
            try:
                res = requests.post(f"{API_URL}/api/setup")
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Banco re-inicializado! Motor ativo: **{data['engine']}**")
                    st.rerun()
                else:
                    st.error(f"Falha ao re-configurar: {res.text}")
            except Exception as e:
                st.error(f"Erro: {e}")

    with col_ingest:
        st.subheader("➕ Ingestão Individual de Produto")
        with st.form("ingest_form"):
            prod_id = st.text_input("ID do Produto (ex: prod_001)", value="prod_")
            prod_name = st.text_input("Nome do Produto")
            prod_cat = st.text_input("Categoria (ex: Vestuário, Calçados, Móveis)")
            prod_price = st.number_input("Preço", min_value=0.0, step=10.0)
            prod_url = st.text_input("URL da Imagem (Link público válido)")

            submit_ingest = st.form_submit_button("📥 Ingerir & Gerar Vetor CLIP")

            if submit_ingest:
                if not prod_id or not prod_name or not prod_cat or not prod_url:
                    st.error("Todos os campos são obrigatórios!")
                else:
                    payload = {
                        "id": prod_id,
                        "nome_produto": prod_name,
                        "categoria": prod_cat,
                        "preco": prod_price,
                        "url_imagem": prod_url
                    }
                    with st.spinner("Gerando embedding do CLIP e indexando..."):
                        try:
                            res = requests.post(f"{API_URL}/api/products", json=payload)
                            if res.status_code == 200:
                                st.success("Produto adicionado com sucesso!")
                                st.rerun()
                            else:
                                st.error(f"Falha na indexação: {res.text}")
                        except Exception as e:
                            st.error(f"Erro: {e}")

    # Seed mock dataset button
    st.markdown("### ⚡ Carga Rápida de Demonstração (Seed Mock Dataset)")
    st.write("Insira uma lista de produtos prontos de alta qualidade para testar e demonstrar o sistema instantaneamente.")

    if st.button("🌱 Carregar Dados de Exemplo"):
        mock_data = [
            {
                "id": "cadeira_minimalista",
                "nome_produto": "Cadeira de Escritório Ergonômica Minimalista",
                "categoria": "Móveis",
                "preco": 850.0,
                "url_imagem": "https://images.unsplash.com/photo-1505797149-43b0069ec26b?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "cadeira_designer_preta",
                "nome_produto": "Cadeira Escandinava de Designer Preta",
                "categoria": "Móveis",
                "preco": 320.0,
                "url_imagem": "https://images.unsplash.com/photo-1567538096630-e0c55bd6374c?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "sofa_azul_moderno",
                "nome_produto": "Sofá Modular de Veludo Azul",
                "categoria": "Móveis",
                "preco": 3400.0,
                "url_imagem": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "tenis_corrida_vermelho",
                "nome_produto": "Tênis Ultralight Pro Vermelho",
                "categoria": "Calçados",
                "preco": 450.0,
                "url_imagem": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "tenis_corrida_azul",
                "nome_produto": "Tênis de Corrida Confort Azul",
                "categoria": "Calçados",
                "preco": 399.9,
                "url_imagem": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "sapato_couro_classico",
                "nome_produto": "Sapato Oxford de Couro Nobre",
                "categoria": "Calçados",
                "preco": 590.0,
                "url_imagem": "https://images.unsplash.com/photo-1533867617858-e7b97e060509?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "vestido_floral_festa",
                "nome_produto": "Vestido Longo Floral para Festa",
                "categoria": "Vestuário",
                "preco": 289.0,
                "url_imagem": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "vestido_casual_vermelho",
                "nome_produto": "Vestido Casual Elegante Vermelho",
                "categoria": "Vestuário",
                "preco": 199.9,
                "url_imagem": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?auto=format&fit=crop&w=600&q=80"
            },
            {
                "id": "relogio_minimalista_couro",
                "nome_produto": "Relógio Clássico pulseira de Couro Marrom",
                "categoria": "Acessórios",
                "preco": 750.0,
                "url_imagem": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?auto=format&fit=crop&w=600&q=80"
            }
        ]

        # Call API for each product
        progress_bar = st.progress(0)
        success_count = 0
        for idx, item in enumerate(mock_data):
            try:
                # Recreate clean start
                if idx == 0:
                    requests.post(f"{API_URL}/api/setup")
                res = requests.post(f"{API_URL}/api/products", json=item)
                if res.status_code == 200:
                    success_count += 1
            except Exception as e:
                st.error(f"Erro ao cadastrar {item['id']}: {e}")
            progress_bar.progress((idx + 1) / len(mock_data))

        st.success(f"Carga de demonstração finalizada! {success_count} de {len(mock_data)} produtos indexados com sucesso.")
        st.rerun()
