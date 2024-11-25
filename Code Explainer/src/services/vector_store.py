from chromadb import Client, Settings
import chromadb
import hashlib
import re

class CodeVectorStore:
    def __init__(self):
        # Initialize ChromaDB with persistent storage
        self.client = Client(Settings(
            persist_directory="./.chromadb",
            anonymized_telemetry=False
        ))
        
        # Create or get our collection
        self.collection = self.client.get_or_create_collection(
            name="code_segments",
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_id(self, content: str) -> str:
        """Generate a stable ID for a piece of content"""
        return hashlib.md5(content.encode()).hexdigest()

    def _split_into_chunks(self, content: str, chunk_size: int = 1000) -> list:
        """Split content into overlapping chunks"""
        chunks = []
        
        # Split content into logical segments (functions, structs, etc.)
        segments = re.split(r'(\n\n|\n(?=[a-zA-Z_]\w*\s+[a-zA-Z_]\w*\s*\()))', content)
        
        current_chunk = ""
        for segment in segments:
            if len(current_chunk) + len(segment) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Keep last part for overlap
                current_chunk = current_chunk[-200:] + segment
            else:
                current_chunk += segment
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def add_file(self, file_path: str, content: str):
        """Add a file's content to the vector store"""
        # Remove existing entries for this file
        self.collection.delete(
            where={"file_path": file_path}
        )
        
        # Split content into chunks
        chunks = self._split_into_chunks(content)
        
        # Prepare documents for insertion
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_id(f"{file_path}:{i}:{chunk}")
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "file_path": file_path,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 5) -> list:
        """Search for relevant code segments"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            formatted_results.append({
                'content': doc,
                'file_path': metadata['file_path'],
                'chunk_index': metadata['chunk_index'],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results 