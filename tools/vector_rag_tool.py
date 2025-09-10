"""
Vector-based Retrieval-Augmented Generation (RAG) tool.

This module implements a proper vector-based RAG system using sentence-transformers
for embeddings and ChromaDB for vector storage, as specified in the project requirements.
It provides domain-specific RAG tools for different business areas.
"""

from __future__ import annotations

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    import chromadb
    from chromadb.config import Settings
    HAS_VECTOR_DEPS = True
except ImportError:
    HAS_VECTOR_DEPS = False
    print("Warning: Vector dependencies not installed. Install with: pip install sentence-transformers chromadb")

import database



import os
from typing import Dict, List, Optional, Any


class VectorRAGTool:
    """Vector-based RAG tool using Google Generative AI embeddings and ChromaDB."""
    BATCH_SIZE = 100  # Default batch size for adding documents
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None

        if HAS_VECTOR_DEPS:
            self._initialize_vector_components()
        else:
            print("Warning: Vector RAG not available. Falling back to basic search.")

    def _initialize_vector_components(self):
        """Initialize ChromaDB client."""
        try:
            # Initialize ChromaDB client
            logging.info("Initializing ChromaDB client...")
            self.chroma_client = chromadb.PersistentClient(
                path="./vector_db",
                settings=Settings(anonymized_telemetry=False)
            )
            logging.info("ChromaDB client initialized.")

            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            # Load documents if collection is empty
            self._load_documents_if_needed()

        except Exception as e:
            print(f"Warning: Failed to initialize vector components: {e}")
            self.chroma_client = None
            self.collection = None

    def _load_documents_if_needed(self):
        """Load documents into vector database if not already present."""
        if not self.collection:
            return

        try:
            count = self.collection.count()
            if count > 0:
                return  # Documents already loaded
        except:
            pass

        print("Loading documents into vector database...")
        self._load_all_documents()

    def _load_all_documents(self):
        """Load all documents from the database into the vector store."""
        if not self.collection:
            return

        rows = database.query("SELECT id, module, path, tags FROM documents", None)

        documents = []
        metadatas = []
        ids = []

        for row in rows:
            doc_id = row["id"]
            module = row["module"]
            path = row["path"]
            tags = row["tags"] or ""

            # Read document content
            text = ""
            if path and os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

            if not text.strip():
                continue

            # Split document into chunks for better retrieval
            chunks = self._split_document(text, path)

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                documents.append(chunk)
                metadatas.append({
                    "doc_id": doc_id,
                    "module": module,
                    "path": path,
                    "tags": tags,
                    "chunk_index": i
                })
                ids.append(chunk_id)

        if documents:
            # Generate embeddings for all chunks
            embeddings = self._generate_embeddings(documents)

            # Add documents to collection in batches
            batch_size = self.BATCH_SIZE
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]

                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_metas,
                        ids=batch_ids,
                        embeddings=batch_embeddings
                    )
                except Exception as e:
                    print(f"Error adding batch {i//batch_size}: {e}")

            print(f"Loaded {len(documents)} document chunks into vector database")

    def _split_document(self, text: str, path: str) -> List[str]:
        """Split document into smaller chunks for better retrieval."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        max_chunk_size = 500  # characters

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        if not chunks and text.strip():
            chunks = [text[:max_chunk_size]]

        return chunks

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using GoogleGenerativeAIEmbeddings."""
        try:
            embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001",)
            return embedder.embed_documents(texts)
        except Exception as e:
            # log traceback
            logging.error(f"Error generating embeddings: {e}")
            # Skip documents with errors instead of returning zero embeddings
            return []

    def search(
        self,
        query: str,
        k: int = 3,
        module: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents using vector similarity."""
        if not self.collection:
            return []

        try:
            # Generate embedding for the query
            query_embedding = self._generate_embeddings([query])[0]

            # Build filter conditions
            where_conditions = {}
            if module:
                where_conditions["module"] = module

            if tags:
                pass

            # Query the vector database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_conditions if where_conditions else None
            )

            if not results["documents"] or not results["documents"][0]:
                return []

            processed_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                score = max(0, 1 - distance)
                processed_results.append({
                    "id": metadata["doc_id"],
                    "module": metadata["module"],
                    "tags": metadata["tags"],
                    "score": score,
                    "excerpt": doc[:200] + "..." if len(doc) > 200 else doc,
                    "path": metadata["path"]
                })

                if len(processed_results) >= k:
                    break

            return processed_results

        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

# Domain-specific RAG tools as required by project specifications
class PolicyRAGTool(VectorRAGTool):
    """Finance policy document search tool."""
    
    def __init__(self):
        super().__init__("policy_documents")
    
    def search_policies(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search specifically in policy documents."""
        return self.search(query, k=k, tags="policy")


class DocRAGTool(VectorRAGTool):
    """General document search tool for supplier contracts and incident reports."""
    
    def __init__(self):
        super().__init__("business_documents")
    
    def search_contracts(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search supplier contracts and procurement documents."""
        return self.search(query, k=k, module="inventory", tags="contracts")
    
    def search_incidents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search incident reports."""
        return self.search(query, k=k, tags="incidents")


class DefinitionRAGTool(VectorRAGTool):
    """Business glossary and metric definition search tool."""
    
    def __init__(self):
        super().__init__("definition_documents")
    
    def search_definitions(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search business definitions and glossary."""
        return self.search(query, k=k, tags="glossary,metrics")


class SalesRAGTool(VectorRAGTool):
    """Sales and CRM document search tool."""
    
    def __init__(self):
        super().__init__("sales_documents")
    
    def search_procedures(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search sales procedures and CRM documentation."""
        return self.search(query, k=k, module="sales")
