import PyPDF2


class PDFHandler:
    """Handler for PDF file operations"""

    def extract_text(self, pdf_path):
        """
        Extract text from a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a string
        """
        text = ""

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()

                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text

        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte du PDF: {str(e)}")

        return text.strip()

    def get_pdf_info(self, pdf_path):
        """
        Get information about a PDF file

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with PDF information
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                info = {
                    'num_pages': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata if pdf_reader.metadata else {}
                }

                return info

        except Exception as e:
            raise Exception(f"Erreur lors de la lecture des informations du PDF: {str(e)}")
