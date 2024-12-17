import PyPDF2

class PDFProcessor:
    def extract_text(self, pdf_path):
        """
        Extracts text from a PDF file.

        :param pdf_path: Path to the PDF file
        :return: Raw text as a string
        """
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
