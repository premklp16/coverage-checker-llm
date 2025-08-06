import sys


try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2


def pdf_to_lines(pdf_list, output_txt_path):
    """
    Reads a PDF file, extracts text line by line, and saves to a text file.
    
    Args:
        input_pdf_path (str): Path to the input PDF file
        output_txt_path (str): Path to save the output text file
    """
    try:
        # Open the PDF file in read-binary mode
        for input_pdf_path in pdf_list:
            with open(input_pdf_path, 'rb') as pdf_file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Initialize an empty string to hold all text
                full_text = ""
                
                # Iterate through each page and extract text
                for page in pdf_reader.pages:
                    full_text += page.extract_text()
                
                # Split the text into lines
                lines = full_text.split('\n')
                
                # Write lines to the output text file
                with open(output_txt_path, 'a', encoding='utf-8') as txt_file:
                    for line in lines:
                        txt_file.write(line + '\n')
            
            print(f"Successfully converted {input_pdf_path} to {output_txt_path}")
            print(f"Total pages: {len(pdf_reader.pages)}")
            print(f"Total lines: {len(lines)}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_pdf_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

 