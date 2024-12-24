import chromadb.utils.embedding_functions as embedding_functions
import os
import uuid
import chromadb
import yaml
import argparse
from dotenv import load_dotenv
load_dotenv()

def load_notes(file_path):
    """Load notes from a text file, where each line is a note."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description='Prepare vector database from notes')
    parser.add_argument('--notes-file', type=str, required=True,
                      help='Path to the text file containing notes (one per line)')
    args = parser.parse_args()

    # Load config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load notes from file
    notes = load_notes(args.notes_file)

    documents = [
        {
            "details": note
        }
        for note in notes
    ]

    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        api_type='azure',
        model_name=config.get('azure_embedding_model')
    )

    chroma_client = chromadb.PersistentClient(path=config.get('db_persist_path'))

    try:
        collection = chroma_client.get_collection(
            config.get('db_collection_name'), 
            embedding_function=ef
        )
    except:
        collection = chroma_client.create_collection(
            config.get('db_collection_name'),
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef
        )
        collection.add(
            documents=[d['details'] for d in documents],
            ids=[str(uuid.uuid4()) for _ in documents],
        )

if __name__ == "__main__":
    main()