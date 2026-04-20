import pdfplumber
import os
import glob

# ================== НАСТРОЙКИ ==================
# Путь к папке с PDF-файлами (измените на свою)
PDF_FOLDER = r'C:\Users\...\Documents\кат'   # папка, где лежат ваши PDF
# Путь для сохранения итогового текстового файла
OUTPUT_FILE = r'C:\Users\...\Documents\кат\full_document.txt'
# ===============================================

def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF-файла"""
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return '\n'.join(text_pages)

# Находим все PDF в папке
pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, '*.pdf')))
if not pdf_files:
    print(f"❌ PDF файлы не найдены в папке: {PDF_FOLDER}")
else:
    all_text = []
    for pdf_path in pdf_files:
        print(f"Обработка: {os.path.basename(pdf_path)}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            all_text.append(text)
        else:
            print(f"  Внимание: не удалось извлечь текст из {os.path.basename(pdf_path)} (возможно, это сканированный PDF)")
    
    full_text = '\n\n'.join(all_text)
    
    # Сохраняем результат
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"\n✅ Текст сохранён в: {OUTPUT_FILE}")
    print(f"Всего символов: {len(full_text)}")
