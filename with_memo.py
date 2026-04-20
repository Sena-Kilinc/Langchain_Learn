from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Model
llm = ChatOllama(model="llama3.2")

# Geçmiş deposu
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Prompt — geçmişe yer açıldı
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımsever bir Türkçe asistansın."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{soru}")
])

chain = prompt | llm

# Memory bağlandı
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="soru",
    history_messages_key="history"
)

# Sohbet fonksiyonu
def chat(soru: str, session_id: str = "kullanici_1"):
    response = chain_with_memory.invoke(
        {"soru": soru},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

# --- Dene ---
print(chat("Merhaba! Benim adım Ahmet."))
print(chat("Benim adım ne?"))          # hatırlaması lazım
print(chat("Peki LangChain nedir?"))