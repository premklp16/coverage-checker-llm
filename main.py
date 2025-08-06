import asyncio
import pdf_to_text
import Embed_the_text_file
import insurance_coverage
import clear_files
uploaded_documents = []
pdf_list = []

def pdfToText():
    while True:
        fileName = input("enter file name (or 'exit' to quit): ").strip()
        if fileName.lower() == 'exit':
            break
        pdf_list.append(fileName)
    output_txt = "policy.txt"
    pdf_to_text.pdf_to_lines(pdf_list, output_txt)
    uploaded_documents.extend(pdf_list)

def TextToJson():
    input_txt = "policy.txt"
    output_json = "embeddings.json"
    Embed_the_text_file.process_text_file(input_txt, output_json, model_name='all-MiniLM-L6-v2', batch_size=64)

if __name__ == "__main__":
    pdfToText()
    TextToJson()
    asyncio.run(insurance_coverage.promptTheQuery())
    clear_files.clear()