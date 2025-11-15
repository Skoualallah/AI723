import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGHandler:
    """Handler for RAG (Retrieval-Augmented Generation) operations"""

    def __init__(self, embeddings_file="rag_embeddings.json"):
        self.embeddings_file = embeddings_file
        self.model = None
        self.chunks_data = []  # Liste de {filename, chunk_index, text, embedding}

    def load_model(self):
        """Load the sentence transformer model (lazy loading)"""
        if self.model is None:
            # Using a lightweight multilingual model
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self.model

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """
        Split text into overlapping chunks

        Args:
            text: Text to split
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def process_pdf(self, filename, content, chunk_size=500, overlap=50):
        """
        Process a PDF by creating chunks and embeddings

        Args:
            filename: PDF filename
            content: Extracted text content from PDF
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks

        Returns:
            Number of chunks created
        """
        model = self.load_model()

        # Split into chunks
        chunks = self.chunk_text(content, chunk_size, overlap)

        # Create embeddings for each chunk
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk, convert_to_numpy=True)

            # Store chunk data
            self.chunks_data.append({
                'filename': filename,
                'chunk_index': i,
                'text': chunk,
                'embedding': embedding.tolist()  # Convert to list for JSON serialization
            })

        return len(chunks)

    def remove_pdf(self, filename):
        """
        Remove all chunks for a specific PDF

        Args:
            filename: PDF filename to remove
        """
        self.chunks_data = [
            chunk for chunk in self.chunks_data
            if chunk['filename'] != filename
        ]

    def search(self, query, top_k=5):
        """
        Search for the most relevant chunks based on a query

        Args:
            query: User query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with their similarity scores
        """
        if not self.chunks_data:
            return []

        model = self.load_model()

        # Create embedding for the query
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Calculate similarity with all chunks
        similarities = []
        for chunk in self.chunks_data:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk_embedding.reshape(1, -1)
            )[0][0]

            similarities.append({
                'filename': chunk['filename'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'],
                'similarity': float(similarity)
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Return top k results
        return similarities[:top_k]

    def build_rag_context(self, query, top_k=5):
        """
        Build context from RAG search results

        Args:
            query: User query
            top_k: Number of chunks to include

        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)

        if not results:
            return ""

        context_parts = []
        context_parts.append("=== Base de Connaissances (RAG) ===\n")

        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source: {result['filename']} - Chunk {result['chunk_index']}]")
            context_parts.append(f"[Pertinence: {result['similarity']:.2%}]")
            context_parts.append(result['text'])
            context_parts.append("")

        context_parts.append("=== Fin de la Base de Connaissances ===\n")

        return "\n".join(context_parts)

    def save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self):
        """Load embeddings from file"""
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    self.chunks_data = json.load(f)
                return True
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                return False
        return False

    def get_stats(self):
        """Get statistics about the RAG system"""
        if not self.chunks_data:
            return {
                'total_chunks': 0,
                'total_pdfs': 0,
                'pdfs': {}
            }

        # Count chunks per PDF
        pdfs = {}
        for chunk in self.chunks_data:
            filename = chunk['filename']
            if filename not in pdfs:
                pdfs[filename] = 0
            pdfs[filename] += 1

        return {
            'total_chunks': len(self.chunks_data),
            'total_pdfs': len(pdfs),
            'pdfs': pdfs
        }

    def clear_all(self):
        """Clear all embeddings"""
        self.chunks_data = []
        if os.path.exists(self.embeddings_file):
            os.remove(self.embeddings_file)
