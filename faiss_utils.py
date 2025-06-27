import os
import pickle
import logging
from typing import List, Tuple, Optional

try:
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install the required packages:")
    print("pip install faiss-cpu numpy scikit-learn")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorahVectorDB:
    """
    Singleton FAISS vector database for Torah documents using TF-IDF embeddings.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, chunk_size: int = 512, max_features: int = 10000, tokenizer=None):
        if cls._instance is None:
            cls._instance = super(TorahVectorDB, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, chunk_size: int = 512, max_features: int = 10000, tokenizer=None):
        """
        Initialize the Torah Vector Database.
        
        Args:
            chunk_size (int): Maximum size of document chunks. If tokenizer is provided, 
                             this is in tokens; otherwise in characters. Defaults to 512.
            max_features (int): Maximum number of TF-IDF features to use. Defaults to 10000.
            tokenizer: Optional tokenizer for token-based chunking. If None, uses character-based chunking.
        """
        if TorahVectorDB._initialized:
            return
            
        self.chunk_size = chunk_size
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.use_token_chunking = tokenizer is not None
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        self.document_chunks = []  # Store the actual text chunks
        self.chunk_metadata = []   # Store metadata about each chunk (file, position, etc.)
        self.index = None  # Will be initialized after vectorization
        self.embedding_dim = None  # Will be set after vectorization
        
        # Load documents and build index
        self._load_and_index_documents()
        
        TorahVectorDB._initialized = True
        logger.info(f"TorahVectorDB initialized with {len(self.document_chunks)} chunks")
    
    def _load_and_index_documents(self):
        """
        Load all documents from torah-cqa/contexts and index them.
        """
        contexts_dir = "torah-cqa/contexts"
        
        if not os.path.exists(contexts_dir):
            logger.error(f"Contexts directory not found: {contexts_dir}")
            return
        
        text_files = [f for f in os.listdir(contexts_dir) if f.endswith('.txt')]
        logger.info(f"Found {len(text_files)} text files to process")
        
        all_chunks = []
        
        for filename in text_files:
            filepath = os.path.join(contexts_dir, filename)
            logger.info(f"Processing {filename}...")
            
            chunks = self._chunk_document(filepath, filename)
            if chunks:
                all_chunks.extend(chunks)
        
        if all_chunks:
            # Store chunks and metadata
            self.document_chunks = [chunk['text'] for chunk in all_chunks]
            self.chunk_metadata = [
                {
                    'filename': chunk['filename'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos'],
                    'chunk_id': i
                }
                for i, chunk in enumerate(all_chunks)
            ]
            
            # Vectorize all chunks
            logger.info("Creating TF-IDF embeddings...")
            tfidf_matrix = self.vectorizer.fit_transform(self.document_chunks)
            embeddings = tfidf_matrix.toarray().astype('float32')
            
            # Set embedding dimension
            self.embedding_dim = embeddings.shape[1]
            
            # Initialize FAISS index and add embeddings
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero
            self.index.add(embeddings)
            
            logger.info(f"Successfully indexed {len(all_chunks)} chunks from {len(text_files)} files")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _chunk_document(self, filepath: str, filename: str) -> List[dict]:
        """
        Chunk a document into smaller pieces based on chunk_size.
        
        Args:
            filepath (str): Path to the document file
            filename (str): Name of the file for metadata
            
        Returns:
            List[dict]: List of chunk dictionaries with text and metadata
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return []
        
        chunks = []
        
        if self.use_token_chunking and self.tokenizer is not None:
            # Token-based chunking
            tokens = self.tokenizer.encode(content)
            total_tokens = len(tokens)
            
            for start in range(0, total_tokens, self.chunk_size):
                end = min(start + self.chunk_size, total_tokens)
                chunk_tokens = tokens[start:end]
                
                # Skip empty chunks
                if not chunk_tokens:
                    continue
                
                # Decode tokens back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
                
                # Skip empty text chunks
                if not chunk_text:
                    continue
                
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'start_pos': start,  # Token position
                    'end_pos': end       # Token position
                })
        else:
            # Character-based chunking (original logic)
            content_length = len(content)
            
            for start in range(0, content_length, self.chunk_size):
                end = min(start + self.chunk_size, content_length)
                chunk_text = content[start:end].strip()
                
                # Skip empty chunks
                if not chunk_text:
                    continue
                
                # Try to break at sentence boundaries when possible
                if end < content_length and not chunk_text.endswith(('.', '!', '?', '\n')):
                    # Look for the last sentence boundary in the chunk
                    last_sentence_end = max(
                        chunk_text.rfind('.'),
                        chunk_text.rfind('!'),
                        chunk_text.rfind('?'),
                        chunk_text.rfind('\n')
                    )
                    if last_sentence_end > len(chunk_text) * 0.7:  # Only if we don't lose too much text
                        chunk_text = chunk_text[:last_sentence_end + 1].strip()
                        end = start + last_sentence_end + 1
                
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'start_pos': start,
                    'end_pos': end
                })
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def query(self, question: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Query the vector database for the most relevant document chunks.
        
        Args:
            question (str): The question/query to search for
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float, dict]]: List of tuples containing:
                - document chunk text
                - similarity score
                - metadata dictionary
        """
        if not self.document_chunks or self.index is None:
            logger.warning("No documents indexed yet")
            return []
        
        # Vectorize the question using the same TF-IDF vectorizer
        question_tfidf = self.vectorizer.transform([question])
        question_embedding = question_tfidf.toarray().astype('float32')
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(question_embedding, axis=1, keepdims=True)
        question_embedding = question_embedding / (norm + 1e-8)
        
        # Search the index
        top_k = min(top_k, len(self.document_chunks))  # Don't request more than available
        scores, indices = self.index.search(question_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.document_chunks):  # Safety check
                results.append((
                    self.document_chunks[idx],
                    float(score),
                    self.chunk_metadata[idx].copy()
                ))
        
        return results
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector database.
        
        Returns:
            dict: Statistics including number of chunks, files processed, etc.
        """
        file_counts = {}
        for metadata in self.chunk_metadata:
            filename = metadata['filename']
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        return {
            'total_chunks': len(self.document_chunks),
            'total_files': len(file_counts),
            'chunks_per_file': file_counts,
            'embedding_dimension': self.embedding_dim,
            'chunk_size': self.chunk_size,
            'max_features': self.max_features
        }
    
    def save_index(self, filepath: str = "torah_vector_db.pkl"):
        """
        Save the vector database to disk.
        
        Args:
            filepath (str): Path where to save the index
        """
        try:
            # Save FAISS index
            faiss_index_path = filepath.replace('.pkl', '.faiss')
            faiss.write_index(self.index, faiss_index_path)
            
            # Save other data
            data_to_save = {
                'document_chunks': self.document_chunks,
                'chunk_metadata': self.chunk_metadata,
                'chunk_size': self.chunk_size,
                'max_features': self.max_features,
                'embedding_dim': self.embedding_dim,
                'vectorizer': self.vectorizer
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            logger.info(f"Vector database saved to {filepath} and {faiss_index_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def load_index(self, filepath: str = "torah_vector_db.pkl"):
        """
        Load the vector database from disk.
        
        Args:
            filepath (str): Path to load the index from
        """
        try:
            # Load FAISS index
            faiss_index_path = filepath.replace('.pkl', '.faiss')
            if os.path.exists(faiss_index_path):
                self.index = faiss.read_index(faiss_index_path)
            
            # Load other data
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.document_chunks = data['document_chunks']
                self.chunk_metadata = data['chunk_metadata']
                self.chunk_size = data['chunk_size']
                self.max_features = data['max_features']
                self.embedding_dim = data['embedding_dim']
                self.vectorizer = data['vectorizer']
                
                logger.info(f"Vector database loaded from {filepath}")
                return True
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
        
        return False


# Global instance functions for easy access
_global_db = None

def get_torah_vector_db(chunk_size: int = 512, max_features: int = 10000, tokenizer=None) -> TorahVectorDB:
    """
    Get the global Torah vector database instance.
    
    Args:
        chunk_size (int): Maximum size of document chunks (tokens if tokenizer provided, characters otherwise)
        max_features (int): Maximum number of TF-IDF features
        tokenizer: Optional tokenizer for token-based chunking
        
    Returns:
        TorahVectorDB: The singleton database instance
    """
    global _global_db
    if _global_db is None:
        _global_db = TorahVectorDB(chunk_size, max_features, tokenizer)
    return _global_db

def query_torah_db(question: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
    """
    Convenience function to query the global Torah vector database.
    
    Args:
        question (str): The question/query to search for
        top_k (int): Number of top results to return
        
    Returns:
        List[Tuple[str, float, dict]]: List of tuples containing document chunks, scores, and metadata
    """
    db = get_torah_vector_db()
    return db.query(question, top_k)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the database
    print("Initializing Torah Vector Database...")
    db = get_torah_vector_db(chunk_size=512)
    
    # Print statistics
    stats = db.get_stats()
    print(f"\nDatabase Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Chunk size: {stats['chunk_size']}")
    print(f"Max features: {stats['max_features']}")
    print(f"Files processed: {list(stats['chunks_per_file'].keys())}")
    
    # Test query
    print(f"\nTesting query...")
    question = "Who created the heavens and the earth?"
    results = query_torah_db(question, top_k=3)
    
    print(f"\nQuery: '{question}'")
    print(f"Top {len(results)} results:")
    for i, (chunk, score, metadata) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   File: {metadata['filename']}")
        print(f"   Chunk: {chunk[:200]}...")
