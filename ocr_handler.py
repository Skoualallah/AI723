import pytesseract
from PIL import Image, ImageGrab
import os
import platform

class OCRHandler:
    def __init__(self):
        """Initialize OCR handler with Tesseract path for Windows"""
        # Configure Tesseract path for Windows
        if platform.system() == "Windows":
            # Path par défaut de Tesseract sur Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                # Essayer le path en Program Files (x86)
                tesseract_path_x86 = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                if os.path.exists(tesseract_path_x86):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path_x86

    def extract_text_from_image(self, image_path, lang='fra'):
        """
        Extract text from an image file using Tesseract OCR

        Args:
            image_path (str): Path to the image file
            lang (str): Language for OCR (default: 'fra' for French, use 'eng' for English)

        Returns:
            str: Extracted text from the image
        """
        try:
            # Open the image
            image = Image.open(image_path)

            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang)

            return text.strip()
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte: {str(e)}")

    def extract_text_from_clipboard(self, lang='fra'):
        """
        Extract text from an image in the clipboard using Tesseract OCR

        Args:
            lang (str): Language for OCR (default: 'fra' for French, use 'eng' for English)

        Returns:
            str: Extracted text from the clipboard image
        """
        try:
            # Get image from clipboard
            image = ImageGrab.grabclipboard()

            if image is None:
                raise Exception("Aucune image trouvée dans le presse-papier")

            # Check if it's an image
            if not isinstance(image, Image.Image):
                raise Exception("Le contenu du presse-papier n'est pas une image")

            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang)

            return text.strip()
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte depuis le presse-papier: {str(e)}")

    def extract_text_from_pil_image(self, image, lang='fra'):
        """
        Extract text from a PIL Image object

        Args:
            image (PIL.Image): PIL Image object
            lang (str): Language for OCR (default: 'fra' for French, use 'eng' for English)

        Returns:
            str: Extracted text from the image
        """
        try:
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang)

            return text.strip()
        except Exception as e:
            raise Exception(f"Erreur lors de l'extraction du texte: {str(e)}")
