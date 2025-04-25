import streamlit as st
import os
import requests
import json
from bs4 import BeautifulSoup
# from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# load_dotenv()
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="LLM RAG Search", page_icon="🔍")
st.title("🔍 LLM-based RAG Search")
st.caption("Memory-enabled chat. Ask follow-ups!")

#chat history
os.makedirs("chat_history", exist_ok=True)
HISTORY_FILE = "chat_history/user_1.json"

# session state
if "chat_history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            st.session_state.chat_history = json.load(f)
    else:
        st.session_state.chat_history = []

# LangChain setup for memory
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Helper functions, this extract first 2 articles or google search based on our query
def search_articles(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    data = {"q": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    return [{"title": r["title"], "url": r["link"]} for r in results.get("organic", [])[:2]]

def fetch_article_content(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        content = [tag.get_text().strip() for tag in soup.find_all(['h1', 'h2', 'h3', 'p']) if tag.get_text().strip()]
        return "\n".join(content)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def concatenate_content(articles):
    full_text = ""
    for article in articles:
        content = fetch_article_content(article["url"])
        full_text += f"\n\nFrom: {article['title']}\n{content}"
    return full_text.strip()

def generate_answer_with_memory(content, query):
    input_messages = memory.load_memory_variables({})["chat_history"]
    input_messages.append(HumanMessage(content=f"You are a helpful assistant. Use the following content to answer:\n\n{content}\n\nUser question:\n{query}"))
    response = chat_model.invoke(input_messages)
    memory.save_context({"input": query}, {"output": response.content})
    return response.content.strip()

# Clear chat, button iy you want to start with another topic, using clear chat will also delete json from chat_history folder
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.rerun()

# past chat
st.markdown("### 💬 Chat History")
for msg in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {msg['query']}")
    st.markdown(f"**🤖 AI:** {msg['answer']}")

st.markdown("---")
query = st.text_input("Ask something (initial or follow-up):", key="input_query")

if st.button("📤 Send") and query.strip():
    with st.spinner("Thinking..."):
        try:
            articles = search_articles(query)
            content = concatenate_content(articles)
            answer = generate_answer_with_memory(content, query)

            # Saving  & updating
            new_entry = {"query": query, "answer": answer}
            st.session_state.chat_history.append(new_entry)

            with open(HISTORY_FILE, "w") as f:
                json.dump(st.session_state.chat_history, f, indent=2)

            # results
            st.success("### ✅ Answer")
            st.write(answer)

            if articles:
                st.info("### 🔗 Sources")
                for s in articles:
                    st.markdown(f"- [{s['title']}]({s['url']})")

        except Exception as e:
            st.error(f"❌ Error: {e}")
