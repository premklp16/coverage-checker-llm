import json
import os
import sys
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_path = f"local_models/{model_name}"
        try:
            if os.path.exists(self.model_path):
                self.model = SentenceTransformer(self.model_path)
            else:
                print("Model not found locally. Downloading...")
                self.model = SentenceTransformer(model_name)
                os.makedirs("local_models", exist_ok=True)
                self.model.save(self.model_path)
        except Exception as e:
            print(f"ERROR: Failed to initialize model - {str(e)}")
            sys.exit(1)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True).tolist()

def process_text_file(input_file: str, output_file: str, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32) -> None:
    try:
        embedder = TextEmbedder(model_name)
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            raise ValueError("Input file is empty")
        results = []
        progress = tqdm(total=len(lines), desc="Embedding lines", unit="line")
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            embeddings = embedder.embed_texts(batch)
            results.extend({"line": line, "embedding": embedding} for line, embedding in zip(batch, embeddings))
            progress.update(len(batch))
        progress.close()
        with open(output_file, 'w', encoding='utf-8') as f:  # Overwrite instead of append
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if 'results' in locals() and results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)