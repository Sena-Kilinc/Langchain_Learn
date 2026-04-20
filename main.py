from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. MODELİ TANIMLA (ÜCRETSİZ VE LOKAL)
# base_url varsayılan olarak http://localhost:11434 gelir, değiştirmene gerek yok.
llm = ChatOllama(
    model="llama3.2",          # Az önce indirdiğin modelin adı
    temperature=0.7,           # Yaratıcılık seviyesi (0 = ciddi, 1 = hayalperest)
    # num_predict=256          # (İsteğe bağlı) Max token sayısı
)

# 2. PROMPT ŞABLONU OLUŞTUR
# Sisteme bir rol verip, kullanıcıdan gelen girdiyi {input} içine yerleştireceğiz.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardımsever bir yapay zeka asistanısın. Cevaplarını mutlaka Türkçe ver. Teknik terimleri basitleştirerek anlat."),
    ("human", "{input}")
])

# 3. ZİNCİRİ KUR (Prompt + Model)
chain = prompt | llm

# 4. ÇALIŞTIR
print("LangChain + Ollama (Ücretsiz Sohbet)")
print("-" * 40)

while True:
    soru = input("\nSorunuz (Çıkmak için 'q'): ")
    if soru.lower() == 'q':
        break
    
    # Burada chain.invoke ile sorguyu gönderiyoruz.
    # LangChain arka planda Ollama'ya HTTP isteği atıyor, API Key yok.
    try:
        response = chain.invoke({"input": soru})
        print(f"\nAI: {response.content}")
    except Exception as e:
        print(f"Hata oluştu: Ollama çalışıyor mu? {e}")
