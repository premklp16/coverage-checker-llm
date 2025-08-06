import json
import os
import sys
from tqdm import tqdm
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with local model fallback"""
        self.model_path = f"local_models/{model_name}"
        
        try:
            if os.path.exists(self.model_path):
               # print(f"Loading local model from {self.model_path}")
                self.model = SentenceTransformer(self.model_path)
            else:
                print("Model not found locally. Attempting download...")
                self.model = SentenceTransformer(model_name)
                os.makedirs("local_models", exist_ok=True)
                self.model.save(self.model_path)
                
          #  print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except Exception as e:
            # print(f"\nERROR: Failed to initialize model - {str(e)}")
            # print("\nSOLUTION: Please manually download the model:")
            # print(f"1. Visit https://huggingface.co/sentence-transformers/{model_name}")
            # print(f"2. Download the model files")
            # print(f"3. Create folder 'local_models/{model_name}'")
            # print(f"4. Place all model files in that folder")
            sys.exit(1)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Convert texts to embeddings"""
        return self.model.encode(texts,
                              batch_size=batch_size,
                              show_progress_bar=False,
                              convert_to_numpy=True).tolist()

def process_text_file(input_file: str, 
                    output_file: str, 
                    model_name: str = 'all-MiniLM-L6-v2',
                    batch_size: int = 32) -> None:
    """
    Process a text file to create line embeddings and save as JSON
    """
    try:
      #  print("\nInitializing text embedder...")
        embedder = TextEmbedder(model_name)
        
        print(f"\nReading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
      #  print(f"\nProcessing {len(lines)} lines...")
        results: List[Dict[str, object]] = []
        
        progress = tqdm(total=len(lines), desc="Embedding lines", unit="line")
        
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            try:
                embeddings = embedder.embed_texts(batch)
                for line, embedding in zip(batch, embeddings):
                    results.append({
                        "line": line,
                        "embedding": embedding
                    })
                progress.update(len(batch))
            except Exception as e:
                print(f"\nError processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        progress.close()
        
        print(f"\nSaving results to {output_file}")
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
      #  print("\nCompleted successfully!")
       # print(f"Processed {len(results)}/{len(lines)} lines")
     #   print(f"Embedding dimension: {len(results[0]['embedding']) if results else 0}")
    
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        if 'results' in locals() and len(results) > 0:
            print("Partial results will be saved...")
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(results)} embeddings to {output_file}")


    