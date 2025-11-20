import os


class FileHandler:
    """Handler for various file types in the knowledge base"""

    # Supported file types with their extensions and emojis
    SUPPORTED_TYPES = {
        'text': {
            'extensions': ['.txt', '.text'],
            'emoji': '📄',
            'name': 'Fichier Texte'
        },
        'markdown': {
            'extensions': ['.md', '.markdown'],
            'emoji': '📝',
            'name': 'Markdown'
        },
        'python': {
            'extensions': ['.py'],
            'emoji': '🐍',
            'name': 'Python'
        },
        'javascript': {
            'extensions': ['.js', '.jsx', '.ts', '.tsx'],
            'emoji': '📜',
            'name': 'JavaScript/TypeScript'
        },
        'json': {
            'extensions': ['.json'],
            'emoji': '📋',
            'name': 'JSON'
        },
        'csv': {
            'extensions': ['.csv'],
            'emoji': '📊',
            'name': 'CSV'
        },
        'xml': {
            'extensions': ['.xml'],
            'emoji': '🗂️',
            'name': 'XML'
        },
        'html': {
            'extensions': ['.html', '.htm'],
            'emoji': '🌐',
            'name': 'HTML'
        },
        'css': {
            'extensions': ['.css', '.scss', '.sass'],
            'emoji': '🎨',
            'name': 'CSS'
        },
        'code': {
            'extensions': ['.java', '.c', '.cpp', '.h', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt'],
            'emoji': '💻',
            'name': 'Code Source'
        },
        'config': {
            'extensions': ['.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg'],
            'emoji': '⚙️',
            'name': 'Configuration'
        },
        'shell': {
            'extensions': ['.sh', '.bash', '.zsh', '.fish'],
            'emoji': '🖥️',
            'name': 'Script Shell'
        },
        'sql': {
            'extensions': ['.sql'],
            'emoji': '🗄️',
            'name': 'SQL'
        },
        'log': {
            'extensions': ['.log'],
            'emoji': '📈',
            'name': 'Log'
        }
    }

    def get_file_type(self, filename):
        """
        Determine the file type based on extension

        Args:
            filename: Name of the file

        Returns:
            Dictionary with file type information or None if unsupported
        """
        ext = os.path.splitext(filename)[1].lower()

        for file_type, info in self.SUPPORTED_TYPES.items():
            if ext in info['extensions']:
                return {
                    'type': file_type,
                    'emoji': info['emoji'],
                    'name': info['name'],
                    'extension': ext
                }

        return None

    def extract_text(self, file_path):
        """
        Extract text from a file based on its type

        Args:
            file_path: Path to the file

        Returns:
            Extracted text as a string
        """
        filename = os.path.basename(file_path)
        file_info = self.get_file_type(filename)

        if not file_info:
            raise Exception(f"Type de fichier non supporté: {os.path.splitext(filename)[1]}")

        try:
            # For all text-based files, we can use simple text reading
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            if not text.strip():
                raise Exception("Le fichier semble vide")

            return text.strip()

        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return text.strip()
            except Exception as e:
                raise Exception(f"Erreur d'encodage lors de la lecture du fichier: {str(e)}")

        except Exception as e:
            raise Exception(f"Erreur lors de la lecture du fichier: {str(e)}")

    def get_file_info(self, file_path):
        """
        Get information about a file

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        try:
            filename = os.path.basename(file_path)
            file_stats = os.stat(file_path)
            file_type_info = self.get_file_type(filename)

            info = {
                'filename': filename,
                'size': file_stats.st_size,
                'size_kb': round(file_stats.st_size / 1024, 2),
                'type_info': file_type_info
            }

            return info

        except Exception as e:
            raise Exception(f"Erreur lors de la lecture des informations du fichier: {str(e)}")

    def get_supported_extensions_string(self):
        """
        Get a string of all supported extensions for file dialog

        Returns:
            String with all supported extensions
        """
        all_extensions = []
        for type_info in self.SUPPORTED_TYPES.values():
            all_extensions.extend(type_info['extensions'])

        return ' '.join([f'*{ext}' for ext in all_extensions])

    def get_filetypes_for_dialog(self):
        """
        Get filetypes list for tkinter file dialog

        Returns:
            List of tuples for filetypes parameter
        """
        filetypes = [
            ("Tous les fichiers supportés", self.get_supported_extensions_string())
        ]

        # Add individual categories
        for type_info in self.SUPPORTED_TYPES.values():
            ext_pattern = ' '.join([f'*{ext}' for ext in type_info['extensions']])
            filetypes.append((type_info['name'], ext_pattern))

        filetypes.append(("Tous les fichiers", "*.*"))

        return filetypes
