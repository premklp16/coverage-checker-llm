import sys
import PyPDF2

def pdf_to_lines(pdf_list, output_txt_path):
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:  # Overwrite
            for input_pdf_path in pdf_list:
                with open(input_pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    full_text = "".join(page.extract_text() for page in pdf_reader.pages)
                    lines = full_text.split('\n')
                    for line in lines:
                        txt_file.write(line + '\n')
                print(f"Converted {input_pdf_path} to {output_txt_path}")
    except FileNotFoundError:
        print(f"Error: {input_pdf_path} not found")
    except Exception as e:
        print(f"Error: {str(e)}")