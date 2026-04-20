# main_with_memory.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
# Model
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Prompt (Mesaj Geçmişi için bir yer tutucu ekledik: "history")
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımsever bir Türkçe asistanısın."),
    MessagesPlaceholder(variable_name="history"), # <-- ÖNEMLİ KISIM
    ("human", "{input}")
])

chain = prompt | llm

# Hafıza Yönetimi (Bellekte tutar, program kapanınca silinir)
store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Sohbet Döngüsü
print("Hafızalı Sohbet Başladı (session: user1)")
print("-" * 40)

while True:
    user_input = input("\nSen: ")
    if user_input.lower() == 'q':
        break
    
    response = conversation.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user1"}}
    )
    print(f"Bot: {response.content}")
