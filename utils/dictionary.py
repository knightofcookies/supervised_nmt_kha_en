import PyPDF2


def extract_text_from_pdf(pdf_path, txt_path):
    """
    Extracts text from all pages of a PDF file and saves it to a text file.

    Args:
        pdf_path (str): The path to the PDF file.
        txt_path (str): The path to the output text file.
    """
    try:
        with open(pdf_path, "rb") as pdf_file, open(
            txt_path, "w", encoding="utf-8"
        ) as txt_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                txt_file.write(text)

            print(f"Text extracted and saved to {txt_path}")
    except FileNotFoundError:
        print(f"Error: File not found: {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    pdf_file_path = "../datasets/Khasi-English Dictionary.pdf"
    txt_file_path = "../datasets/Khasi-English Dictionary.txt"

    extract_text_from_pdf(pdf_file_path, txt_file_path)
