# Recomendação Visual e Busca Multimodal (Visual & Multimodal Search)

Este projeto implementa um sistema de recomendação visual e busca multimodal (Image-to-Image, Text-to-Image e Product Recommendation) pronto para produção. Ele resolve o clássico problema do e-commerce de encontrar itens difíceis de descrever em palavras através de espaços vetoriais compartilhados de Inteligência Artificial.

## 🏗️ Como Funciona?

O sistema utiliza o modelo multimodal **OpenAI CLIP** (`openai/clip-vit-base-patch32`) para projetar imagens e descrições de texto em um espaço vetorial comum de **512 dimensões**.
A proximidade semântica entre estes vetores é calculada em tempo real para encontrar itens visualmente semelhantes com extrema precisão.

```
 ┌────────────────┐     ┌──────────────────┐
 │ Imagem do Item │ ──> │ Model Image-Enc  │ ──┐
 └────────────────┘     └──────────────────┘   │
                                               ▼
                                      ┌──────────────────┐     ┌─────────────────────┐
                                      │ Vetor (Embedding)│ ──> │ OpenSearch Index    │
                                      │  ex: [0.12, ...] │     │ (k-NN / HNSW)       │
                                      └──────────────────┘     └─────────────────────┘
                                               ▲
 ┌────────────────┐     ┌──────────────────┐   │
 │ Texto ou Foto  │ ──> │ Model Text-Enc   │ ──┘
 │ do Usuário     │     │ (CLIP)           │
 └────────────────┘     └──────────────────┘
```

O sistema foi desenhado de forma resiliente de ponta a ponta:
1. **OpenSearch HNSW KNN Index**: Utiliza o motor de busca vetorial nativo HNSW com similaridade de cosseno (`cosinesimil`) quando disponível.
2. **Local NumPy Fallback Engine**: Se a instância do OpenSearch estiver offline ou inacessível, o sistema ativa automaticamente um motor de processamento vetorial resiliente utilizando NumPy para calcular a similaridade de cosseno por produtos de matrizes normalizadas (L2), garantindo que a aplicação e a interface continuem plenamente operacionais.

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12** / **FastAPI**: Backend robusto para o processamento de imagens e endpoints da API REST.
- **PyTorch** & **Hugging Face Transformers**: Para inferência em tempo real utilizando o modelo CLIP.
- **Pillow (PIL)**: Processamento e normalização das imagens de entrada.
- **OpenSearch** (`opensearch-py`): Indexação e busca vetorial baseada no algoritmo HNSW.
- **NumPy**: Motor matemático local de fallback para cálculo resiliente de similaridade por cosseno.
- **Streamlit**: Dashboard interativo e responsivo para demonstração das capacidades em tempo real.

---

## 📂 Estrutura do Projeto

```
visual-recommendation-search/
├── backend/
│   ├── embeddings.py       # Inferência de embeddings do CLIP (imagens e textos)
│   ├── search_client.py    # Cliente OpenSearch com motor local de fallback em NumPy
│   └── main.py             # FastAPI REST endpoints
├── frontend/
│   └── app.py              # Interface Streamlit interativa e intuitiva
├── tests/
│   ├── test_search.py      # Testes de unidade do cliente de busca e gerador de vetores
│   └── test_api.py         # Testes de integração dos endpoints FastAPI
└── README.md               # Documentação completa
```

---

## 🚀 Como Executar

### 📦 Instalação das Dependências

Instale todos os pacotes necessários:
```bash
pip install torch transformers pillow opensearch-py fastapi uvicorn pydantic requests numpy streamlit pytest httpx python-multipart
```

### 🏃‍♂️ Iniciando o Backend (FastAPI)

Inicie o servidor de desenvolvimento do FastAPI na porta 8000:
```bash
uvicorn visual-recommendation-search.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

A documentação interativa Swagger estará disponível em: [http://localhost:8000/docs](http://localhost:8000/docs)

### 🖥️ Iniciando o Frontend (Streamlit)

Em um terminal separado, execute a interface visual:
```bash
streamlit run visual-recommendation-search/frontend/app.py --server.port 8501
```

Acesse o portal do usuário em: [http://localhost:8501](http://localhost:8501)

### 🧪 Executando os Testes

Para garantir o funcionamento completo do sistema, execute os testes com:
```bash
PYTHONPATH=visual-recommendation-search/backend python3 -m pytest visual-recommendation-search/tests/
```

---

## 🌟 Principais Recursos Demonstrados

1. **Buscar por Foto (Visual Search Pura)**: Faça upload de qualquer imagem e veja o sistema retornar instantaneamente produtos visualmente semelhantes do catálogo.
2. **Produtos Semelhantes (Recomendação no Rodapé)**: Selecione um produto e veja recomendações de "Você também pode gostar disso" baseadas puramente na semelhança estética vetorial.
3. **Busca Híbrida / Zero-Shot**: Escreva termos complexos em português ou inglês e encontre imagens sem a necessidade de tags manuais ou metadados cadastrados.
4. **Resiliência e Fallback**: Sem banco de dados externo ou OpenSearch executando localmente? Sem problemas. O backend ativa um motor local em NumPy inteligente para manter a busca ativa sem interrupções.
