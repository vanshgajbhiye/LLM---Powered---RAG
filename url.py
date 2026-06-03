import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("RAG Assistant")

# ── All your original logic, unchanged ──────────────────────────────────────

@st.cache_resource
def load_rag():
    embedding_model = MistralAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(model="mistral-small-latest")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful AI assistant.
Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document." """),
        ("human",
         """Context:
{context}

Question:
{question}""")
    ])

    return retriever, llm, prompt

retriever, llm, prompt = load_rag()

# ── Chat UI ──────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context…"):
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

        st.markdown(response.content)
        st.caption(f"📄 {len(docs)} chunks retrieved via MMR")

    st.session_state.messages.append({"role": "assistant", "content": response.content})