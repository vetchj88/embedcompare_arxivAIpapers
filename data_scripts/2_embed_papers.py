# 2_embed_papers.py
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODELS = {
    "bge_large": "BAAI/bge-large-en-v1.5",
    "gte_large": "thenlper/gte-large",
    "minilm": "all-MiniLM-L6-v2"
}
CSV_FILE = 'papers_metadata.csv'
DB_PATH = "./chroma_db"

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=DB_PATH)

# --- Load Paper Metadata ---
df = pd.read_csv(CSV_FILE)
# Convert string representation of list back to list
df['authors'] = df['authors'].apply(eval)

# --- THE FIX: Convert author lists to comma-separated strings for ChromaDB ---
df['authors'] = df['authors'].apply(lambda authors: ', '.join(authors))

documents = df['abstract'].tolist()
metadatas = df.to_dict('records')
ids = [str(i) for i in range(len(df))]

# --- Loop Through Models, Embed, and Store ---
for model_name, model_id in MODELS.items():
    print(f"Processing model: {model_name} ({model_id})")

    # Load the embedding model
    model = SentenceTransformer(model_id)

    # Create or get the collection
    collection_name = f"papers_{model_name}"
    collection = client.get_or_create_collection(name=collection_name)

    # Generate embeddings
    embeddings = model.encode(documents, show_progress_bar=True)

    # Store in ChromaDB
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully created and populated collection: {collection_name}\n")

print("All models have been processed and data is stored in ChromaDB.")