import os
import pickle
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ================== КОНФИГУРАЦИЯ ==================
TEXT_FILE_PATH = r'C:\Users\...\Documents\кат\full_document.txt'
SAVE_DIR = r'C:\Users\...\Desktop\rag_data'
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
DEFAULT_TOP_K = 5

# Настройки LM Studio (OpenAI-совместимый API)
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "not-needed"  # LM Studio не требует ключа

# Выберите модель (точное имя из LM Studio)
# Посмотреть список: lms ls
LM_STUDIO_MODEL = "openai/gpt-oss-20b"  # или "qwen/qwen2.5-coder-14b", "google/gemma-2-27b-it"
# ================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# Инициализация клиента LM Studio
client = OpenAI(
    base_url=LM_STUDIO_URL,
    api_key=LM_STUDIO_API_KEY
)

# Шаг 1-5. Создание индекса (только при первом запуске)
if not os.path.exists(os.path.join(SAVE_DIR, 'faiss_index.bin')):
    print("📖 Шаг 1. Загрузка текста...")
    with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Загружено символов: {len(text)}")
    
    print("✂️ Шаг 2. Нарезка на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    print(f"Получено чанков: {len(chunks)}")
    
    print("🧠 Шаг 3. Загрузка модели эмбеддингов...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("📊 Шаг 4. Генерация эмбеддингов...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    print(f"Размерность эмбеддингов: {embeddings.shape}")
    
    print("🔍 Шаг 5. Создание FAISS индекса...")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Индекс создан, векторов: {index.ntotal}")
    
    print("💾 Шаг 6. Сохранение на диск...")
    faiss.write_index(index, os.path.join(SAVE_DIR, 'faiss_index.bin'))
    with open(os.path.join(SAVE_DIR, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)
    with open(os.path.join(SAVE_DIR, 'model_name.txt'), 'w') as f:
        f.write(EMBEDDING_MODEL_NAME)
    print(f"Сохранено в: {SAVE_DIR}")
else:
    print("Загрузка сохранённых данных...")
    index = faiss.read_index(os.path.join(SAVE_DIR, 'faiss_index.bin'))
    with open(os.path.join(SAVE_DIR, 'chunks.pkl'), 'rb') as f:
        chunks = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'model_name.txt'), 'r') as f:
        model_name = f.read().strip()
    embedding_model = SentenceTransformer(model_name)
    print(f"Загружено {len(chunks)} чанков, {index.ntotal} векторов")

# ================== ФУНКЦИИ ==================

def search(query, top_k=DEFAULT_TOP_K):
    """Поиск релевантных чанков"""
    q_emb = embedding_model.encode([query])
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    valid = [(scores[0][i], indices[0][i]) for i in range(len(indices[0])) if indices[0][i] != -1]
    retrieved = [chunks[i] for _, i in valid]
    return retrieved, [s for s, _ in valid]

def ask_lm_studio(question, context_chunks, model=LM_STUDIO_MODEL):
    """Отправляет запрос к LM Studio через OpenAI-совместимый API"""
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Ты — ассистент. Отвечай на вопрос, основываясь исключительно на приведённых ниже отрывках из книги. Если отрывки содержат полезную информацию (даже не в виде готового рецепта), используй её и объясни, как это применить. Если информации нет, напиши "Нет данных".
Контекст:
{context}

Вопрос: {question}

Ответ:"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Ошибка при обращении к LM Studio: {e}\n\nПроверьте:\n1. Запущен ли LM Studio\n2. Загружена ли модель\n3. Запущен ли сервер (lms server start)"

def switch_model(model_name):
    """Переключение между моделями (нужно загрузить модель в LM Studio)"""
    global LM_STUDIO_MODEL
    LM_STUDIO_MODEL = model_name
    print(f"Модель переключена на: {model_name}")
    print("Убедитесь, что модель загружена в LM Studio!")

def check_lm_studio():
    """Проверка доступности LM Studio"""
    try:
        response = client.models.list()
        print("✅ LM Studio доступен")
        print("Доступные модели:")
        for model in response.data[:5]:  # покажем первые 5
            print(f"  - {model.id}")
        return True
    except Exception as e:
        print(f"❌ LM Studio недоступен: {e}")
        print("\nРешение:")
        print("1. Запустите LM Studio")
        print("2. Загрузите модель (lms load <model_name>)")
        print("3. Запустите сервер (lms server start)")
        return False

# ================== ИНТЕРАКТИВНЫЙ ЦИКЛ ==================

print("\n" + "="*60)
print(f"🎯 LM Studio RAG система")
print(f"📚 Загружено чанков: {len(chunks)}")
print(f"🤖 Текущая модель: {LM_STUDIO_MODEL}")
print("💡 Команды:")
print("   /models      - показать доступные модели")
print("   /model <имя> - переключить модель (нужно загрузить в LM Studio)")
print("   /topk N      - изменить количество чанков")
print("   /check       - проверить статус LM Studio")
print("   /exit        - выход")
print("="*60)

# Проверяем подключение к LM Studio
check_lm_studio()

while True:
    user_input = input("\n❓ Ваш вопрос или команда: ")
    
    # Обработка команд
    if user_input.lower() in ['/exit', 'выход', 'quit']:
        print("До свидания!")
        break
    elif user_input == '/models':
        try:
            models = client.models.list()
            print("Доступные модели в LM Studio:")
            for m in models.data:
                print(f"  - {m.id}")
        except:
            print("Не удалось получить список моделей. Запущен ли сервер?")
        continue
    elif user_input.startswith('/model'):
        parts = user_input.split()
        if len(parts) > 1:
            new_model = parts[1]
            switch_model(new_model)
        else:
            print("Используйте: /model <имя_модели>")
        continue
    elif user_input.startswith('/topk'):
        try:
            new_k = int(user_input.split()[1])
            DEFAULT_TOP_K = new_k
            print(f"Количество чанков изменено на: {DEFAULT_TOP_K}")
        except:
            print("Используйте: /topk <число>")
        continue
    elif user_input == '/check':
        check_lm_studio()
        continue
    elif user_input.startswith('/'):
        print("Неизвестная команда. Доступны: /model, /topk, /models, /check, /exit")
        continue
    
    # Обычный вопрос
    print("🔍 Поиск релевантных фрагментов...")
    retrieved_chunks, scores = search(user_input, top_k=DEFAULT_TOP_K)
    
    print(f"\n📄 Найдено {len(retrieved_chunks)} фрагментов:")
    for i, (ch, sc) in enumerate(zip(retrieved_chunks, scores), 1):
        print(f"\n--- Фрагмент {i} (релевантность: {sc:.4f}) ---")
        preview = ch[:300].replace('\n', ' ')
        print(preview + ("..." if len(ch) > 300 else ""))
    
    print("\n🤖 Генерация ответа через LM Studio...")
    answer = ask_lm_studio(user_input, retrieved_chunks, model=LM_STUDIO_MODEL)
    print("\n✅ ОТВЕТ:\n", answer)
    print("\n" + "-"*60)
