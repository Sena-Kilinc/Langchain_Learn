from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm=ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımsever bir Türkçe asistansın. Her zaman Türkçe cevap ver."),
    ("human", "{soru}")
])
chain=prompt|llm

response = chain.invoke({"soru": "LangChain nedir, kısaca anlat."})
print(response)