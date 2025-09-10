import sqlite3
import sqlite_vec
import ollama
from sentence_transformers import SentenceTransformer
import time
import os
import glob
import requests
import re
from bs4 import BeautifulSoup
import psutil  # Add this import
from typing import List
import struct

# --- Vector Serialization ---
def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)

# --- Memory Tracking ---
memory_log = []

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def log_memory(stage, details=""):
    """Log memory usage at a specific stage"""
    memory_mb = get_memory_usage()
    timestamp = time.time()
    memory_log.append({
        'stage': stage,
        'memory_mb': memory_mb,
        'timestamp': timestamp,
        'details': details
    })
    print(f"💾 {stage}: {memory_mb:.1f} MB {details}")

def print_memory_summary():
    """Print a summary of memory usage throughout the process"""
    if not memory_log:
        return
    
    print("\n" + "="*50)
    print("📊 MEMORY USAGE SUMMARY")
    print("="*50)
    
    start_memory = memory_log[0]['memory_mb']
    
    for i, entry in enumerate(memory_log):
        delta = entry['memory_mb'] - memory_log[i-1]['memory_mb'] if i > 0 else 0
        print(f"{entry['stage']:<25} {entry['memory_mb']:>8.1f} MB  {delta:>+7.1f} MB  {entry['details']}")
    
    peak_memory = max(log['memory_mb'] for log in memory_log)
    current_memory = memory_log[-1]['memory_mb']
    
    print("-" * 50)
    print(f"{'Peak Memory Used':<25} {peak_memory:>8.1f} MB")
    print(f"{'Current Memory':<25} {current_memory:>8.1f} MB")
    print(f"{'Total Memory Delta':<25} {current_memory - start_memory:>+8.1f} MB")
    print("="*50 + "\n")

# --- Configuration ---
DRY_RUN = False # Set to True to test logic without running the model
TABLE_NAME = "documents"

# EmbeddingGemma (requires Hugging Face access request)
# Visit: https://huggingface.co/google/embeddinggemma-300m
# Run: huggingface-cli login
EMBEDDING_MODEL = 'google/embeddinggemma-300m'  # Google's new EmbeddingGemma model
EMBEDDING_DIMS = 256  # Truncated from 768 for 3x faster processing (Matryoshka learning)

# More efficient and powerful than llama3
LLM_MODEL = 'qwen3:4b'  # 2.5GB, 256K context, rivals much larger models
DOCS_DIR = 'docs/'  # Directory containing scraped documentation
DB_FILE = "vectors_docs.db"

# Global model instance to avoid reloading
EMBEDDING_MODEL_INSTANCE = None

# Documentation URLs to scrape
DOCUMENTATION_URLS = {
    'sqlite_vec_python': 'https://alexgarcia.xyz/sqlite-vec/python.html',
    'sqlite_vec_demo': 'https://raw.githubusercontent.com/asg017/sqlite-vec/main/examples/simple-python/demo.py',
    'embeddinggemma_google_blog': 'https://developers.googleblog.com/en/introducing-embeddinggemma/',
    'huggingface_embeddinggemma': 'https://huggingface.co/google/embeddinggemma-300m',
    'huggingface_embeddinggemma_blog': 'https://huggingface.co/blog/embeddinggemma',
    'qwen3_ollama': 'https://ollama.com/library/qwen3',
    'sentence_transformers': 'https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html'
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_embedding_model():
    """Get or create the global embedding model instance."""
    global EMBEDDING_MODEL_INSTANCE
    if EMBEDDING_MODEL_INSTANCE is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
        log_memory("Model Load", f"({EMBEDDING_MODEL})")
    return EMBEDDING_MODEL_INSTANCE

def init_database():
    """Initialize the SQLite database with sqlite-vec extension."""
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Create virtual table for vector storage
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME} USING vec0(
            text TEXT,
            source TEXT,
            embedding float[{EMBEDDING_DIMS}]
        )
    """)
    conn.commit()
    return conn

def scrape_docs():
    """Simple function to scrape documentation from URLs and save to docs folder."""
    print("📥 Scraping Documentation")
    print("=" * 40)
    
    # Create docs directory if it doesn't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    successful = 0
    total = len(DOCUMENTATION_URLS)
    
    for name, url in DOCUMENTATION_URLS.items():
        print(f"📄 Fetching: {name}")
        print(f"   URL: {url}")
        
        try:
            # Fetch the page
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            # Parse with Beautiful Soup and extract text
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
            
            # Get text content
            text_content = soup.get_text(separator='\n', strip=True)
            
            if not text_content.strip():
                print(f"   ⚠️  No content extracted from {url}")
                continue
            
            # Save to file
            filename = f"{name}.txt"
            filepath = os.path.join(DOCS_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n")
                f.write(f"Fetched: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(text_content)
            
            print(f"   ✅ Saved to: {filename} ({len(text_content)} chars)")
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Error fetching {url}: {e}")
        
        # Be respectful - add delay between requests
        time.sleep(1)
    
    print(f"\n📊 Results: {successful}/{total} documents scraped")
    return successful > 0

def token_based_chunking(text, tokenizer, max_tokens=2048, overlap_tokens=100):
    """
    Token-based chunking using the actual embedding model's tokenizer.
    Much more accurate than word-based chunking for demo purposes.
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]  # No need to chunk
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk tokens
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        
        # Move start position with overlap
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    
    return chunks

def chunk_text(text, model, max_tokens=2048, overlap_tokens=100):
    """Use token-based chunking with the embedding model's tokenizer."""
    return token_based_chunking(text, model.tokenizer, max_tokens, overlap_tokens)

def ingest_docs():
    """Reads documents from docs directory and ingests them into the vector store."""
    log_memory("Demo Start", "")
    
    # Always do fresh ingestion for demo purposes
    if os.path.exists(DB_FILE):
        print("🗑️  Removing existing database for fresh demo run...")
        os.remove(DB_FILE)
    
    # Remove docs folder for completely fresh scraping
    if os.path.exists(DOCS_DIR):
        print("🗑️  Removing existing docs folder for fresh scraping...")
        import shutil
        shutil.rmtree(DOCS_DIR)
    
    log_memory("After Cleanup", "")
        
    print("--- Starting Document Ingestion ---")
    
    # Always scrape since we removed the docs folder
    print("🌐 Scraping fresh documentation...")
    if not scrape_docs():
        print("❌ Failed to scrape documentation.")
        return
    
    log_memory("After Scraping", "")
        
    # Check if docs were scraped successfully
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    if not doc_files:
        print("❌ No documentation files found after scraping.")
        return
    
    # 2. Initialize embedding model
    model = get_embedding_model()
    
    # 3. Initialize database
    conn = init_database()

    print(f"📁 Found {len(doc_files)} documentation files:")
    for file in doc_files:
        print(f"   • {os.path.basename(file)}")
    
    all_chunks = []
    chunk_sources = []
    
    # 4. Process each document file
    for doc_file in doc_files:
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use token-based chunking with the embedding model's tokenizer
            chunks = chunk_text(content, model, max_tokens=2048, overlap_tokens=100)
            source_name = os.path.basename(doc_file).replace('.txt', '')
            
            all_chunks.extend(chunks)
            chunk_sources.extend([source_name] * len(chunks))
            
            print(f"📄 {source_name}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error reading {doc_file}: {e}")
            continue

    if not all_chunks:
        print("❌ No content found to ingest.")
        return

    print(f"📊 Total chunks to process: {len(all_chunks)}")
    log_memory("After Chunking", f"({len(all_chunks)} chunks)")
    
    # 5. Generate embeddings and insert documents
    start_time = time.time()
    batch_size = 10  # Process in batches for better progress tracking
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_sources = chunk_sources[i:i + batch_size]
        
        for j, (chunk, source) in enumerate(zip(batch_chunks, batch_sources)):
            # Generate embedding using proper document prompt and dimension truncation
            # EmbeddingGemma uses specific prompts for optimal performance
            embedding = model.encode_document(chunk, truncate_dim=256)
            
            # Insert into database with source information
            conn.execute(f"""
                INSERT INTO {TABLE_NAME} (rowid, text, source, embedding)
                VALUES (?, ?, ?, ?)
            """, (i + j + 1, chunk, source, serialize_f32(embedding.tolist())))
        
        # Progress indicator
        processed = min(i + batch_size, len(all_chunks))
        print(f"🔄 Processed {processed}/{len(all_chunks)} chunks...")
    
    conn.commit()
    conn.close()
    end_time = time.time()
    
    print(f"✅ Ingestion complete in {end_time - start_time:.2f} seconds.")
    print(f"📈 Average: {len(all_chunks)/(end_time - start_time):.1f} chunks/second")
    print("--- Ingestion Finished ---")
    log_memory("After Storage", "(data saved to SQLite)")
    print("✅ Document ingestion complete!")


def semantic_search_and_query(query_text, top_k=3):
    """Performs semantic search and generates response using local LLM."""
    log_memory("Query Start", f"('{query_text[:30]}...')")
    
    # 1. Get embedding model (reuse existing instance)
    model = get_embedding_model()

    # 2. Connect to database
    conn = sqlite3.connect(DB_FILE)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # 3. Generate query embedding using proper query prompt and dimension truncation
    # EmbeddingGemma uses specific prompts for optimal performance
    query_embedding = model.encode_query(query_text, truncate_dim=256)

    # 4. Find similar documents using sqlite-vec
    start_time = time.time()
    cursor = conn.execute(f"""
        SELECT rowid, text, source, distance
        FROM {TABLE_NAME}
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (serialize_f32(query_embedding.tolist()), top_k))
    
    results = cursor.fetchall()
    end_time = time.time()
    
    if not results:
        print("❌ No relevant documents found.")
        conn.close()
        log_memory("After Vector Search", "0 results")
        return "No relevant documents found."

    print(f"✅ Found {len(results)} relevant chunks in {end_time - start_time:.3f} seconds")
    log_memory("After Vector Search", f"({len(results)} results)")
    
    # 5. Combine top results for context
    contexts = []
    sources = []
    for _, text, source, distance in results:
        contexts.append(text)
        sources.append(f"{source} (distance: {distance:.3f})")
        print(f"📄 Source: {source} | Distance: {distance:.4f}")
    
    combined_context = "\n\n".join(contexts)
    unique_sources = list(set([s.split(' (')[0] for s in sources]))
    
    # 6. Build the prompt with multiple contexts
    prompt = f"""Use the following contexts to answer the question comprehensively.
If you don't know the answer based on the provided contexts, just say that you don't know.

Contexts:
{combined_context}

Question: {query_text}

Answer:"""

    # 7. Get streaming response from LLM
    print(f"\n💡 Answer (sources: {', '.join(unique_sources)}):")
    print("=" * 60)

    if DRY_RUN:
        response_content = "This is a DRY RUN response based on the found contexts."
        print(response_content)
    else:
        print(f"🤖 {LLM_MODEL} is thinking and responding...")
        print()
        
        start_time = time.time()
        
        # Stream the response in real-time
        try:
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,  # Enable streaming!
            )
            
            full_response = ""
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end='', flush=True)  # Print immediately
                    full_response += content
            
            end_time = time.time()
            print(f"\n\n⚡ Response completed in {end_time - start_time:.2f} seconds.")
            response_text = full_response
            
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
            response_text = "Error during LLM response."

    print("=" * 60)
    
    conn.close()
    log_memory("After LLM Response", "")
    
    # Print memory summary after each query
    print_memory_summary()
    
    return response_text

def main():
    """Main function for the enhanced RAG demo."""
    print("🚀 Private RAG Stack - Official Documentation Demo")
    print("=" * 60)
    print("🔒 100% Private | 💰 Zero Cost | 📱 Offline Capable")
    print("📚 Using official docs from Google, Hugging Face, Ollama, and SQLite-vec")
    print()
    
    # Ingest all documentation
    ingest_docs()
    
    # Run demo queries
    run_demo_queries()
    
    # Demo complete - memory summary already printed after last query
    print("\n🏁 Demo Complete!")

def run_demo_queries():
    """Run a series of demo queries to showcase the RAG system."""
    demo_queries = [
        "What makes EmbeddingGemma special for mobile applications?",
        "How do I use SQLite-vec with Python?",
        "What are the key features of Qwen3 model?",
        "How does vector similarity search work?",
        "What are the benefits of using local embeddings?"
    ]
    
    print("🎯 Running demo queries to showcase semantic search capabilities:")
    print()
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*20} Demo Query {i}/{len(demo_queries)} {'='*20}")
        print()
        print("--- Starting Query ---")
        print(f"Query: {query}")
        
        response = semantic_search_and_query(query)
        
        if response:
            print(f"\n--- Response ---")
            print(response)

if __name__ == "__main__":
    main()
