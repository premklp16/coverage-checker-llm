import asyncio
import pdf_to_text
import Embed_the_text_file
import insurance_coverage
import clear_files
uploaded_documents=[]
pdf_list=[]
def pdfToText():
    finish=False
    while not finish:
        fileName=input("enter file name: ")
        if fileName=="exit":
            finish=False
            break
        input_pdf = fileName  # Change this to your PDF file path
        pdf_list.append(fileName)
    
    
    output_txt = "policy.txt"  # Change this to your desired output path    
    pdf_to_text.pdf_to_lines(pdf_list, output_txt)
    uploaded_documents.append(input_pdf)

def TextToJson():
    input_txt = "policy.txt"  # Your input text file
    output_json = "embeddings.json"  # Output JSON file
    model_name = 'all-MiniLM-L6-v2'  # Fast and good for RAG
    batch_size = 64
    Embed_the_text_file.process_text_file(input_txt, output_json, model_name, batch_size)



if __name__ == "__main__":
    
    pdfToText()
    TextToJson()
    asyncio.run(insurance_coverage.promptTheQuery())
    clear_files.clear()