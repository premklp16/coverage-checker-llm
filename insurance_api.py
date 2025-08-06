import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import aiohttp
import asyncio
import atexit
from typing import Dict, List, Tuple
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pdf_to_text
import Embed_the_text_file
import clear_files
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Coverage API")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Working directory set to {os.getcwd()}")

class FixedInsuranceCoverageChecker:
    def __init__(self, embeddings_file: str = "embeddings.json", top_k: int = 3):
        self.embeddings_file = embeddings_file
        self.top_k = top_k  # Assign top_k to instance
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.lines = [x['line'] for x in self.data]
            self.section_names = [x.get('section', f"Section_{i}") for i, x in enumerate(self.data)]
            self.embeddings = np.array([x['embedding'] for x in self.data])
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    async def query_grok_api(self, sections: List[Tuple[str, str, float]], scenario: str) -> Tuple[str, str]:
        logger.info("Querying Grok API with multiple sections")
        try:
            async with aiohttp.ClientSession() as session:
                score = 0
                evidence = []
                coverage_keywords = {'cover': 1.5, 'include': 1.5, 'eligible': 1.5, 'provide': 1.5}
                exclusion_keywords = {'exclu': -1.5, 'not cover': -1.5, 'limit': -0.5}
                for section_text, section_name, sim in sections:
                    section_lower = section_text.lower()
                    for kw, weight in coverage_keywords.items():
                        if kw in section_lower:
                            score += weight * sim
                            evidence.append(f"Positive: '{kw}' in {section_name} (Relevance: {sim:.1%})")
                    for kw, weight in exclusion_keywords.items():
                        if kw in section_lower:
                            score -= abs(weight) * sim
                            evidence.append(f"Negative: '{kw}' in {section_name} (Relevance: {sim:.1%})")
                scenario_lower = scenario.lower()
                for kw, weight in coverage_keywords.items():
                    if kw in scenario_lower:
                        score += weight * 0.5
                for kw, weight in exclusion_keywords.items():
                    if kw in scenario_lower:
                        score -= abs(weight) * 0.5
                decision = "Covered" if score > 0 else "Not Covered"
                explanation = "\n".join(evidence) or "No strong coverage indicators found"
                return decision, explanation
        except Exception as e:
            logger.error(f"Error querying Grok API: {str(e)}")
            return "Not Covered", f"API error: {str(e)}"

    async def get_coverage_decision(self, scenario: str) -> Dict:
        try:
            scenario_embed = self.embedder.encode(scenario)
            sims = cosine_similarity([scenario_embed], self.embeddings)[0]
            top_indices = np.argsort(sims)[-self.top_k:][::-1]
            relevant_sections = [(self.lines[i], self.section_names[i], float(sims[i])) for i in top_indices]
            confidence = float(max(sims)) * 100

            score = 0
            evidence = []
            coverage_keywords = {'cover': 1.5, 'include': 1.5, 'eligible': 1.5, 'provide': 1.5}
            exclusion_keywords = {'exclu': -1.5, 'not cover': -1.5, 'limit': -0.5}
            for section_text, section_name, sim in relevant_sections:
                section_lower = section_text.lower()
                for kw, weight in coverage_keywords.items():
                    if kw in section_lower:
                        score += weight * sim
                        evidence.append(f"Positive: '{kw}' in {section_name} (Relevance: {sim:.1%})")
                for kw, weight in exclusion_keywords.items():
                    if kw in section_lower:
                        score -= abs(weight) * sim
                        evidence.append(f"Negative: '{kw}' in {section_name} (Relevance: {sim:.1%})")

            if abs(score) > 1.5:
                decision = "Covered" if score > 0 else "Not Covered"
                explanation = "\n".join(evidence)
                source = "Local Decision"
            else:
                decision, explanation = await self.query_grok_api(relevant_sections, scenario)
                source = "Grok API"

            return {
                "decision": decision,
                "sections": [(text, name, sim * 100) for text, name, sim in relevant_sections],
                "confidence": confidence,
                "explanation": explanation,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error processing scenario: {str(e)}")
            return {
                "decision": "Error",
                "sections": [],
                "confidence": 0,
                "explanation": f"Error: {str(e)}",
                "source": "Error"
            }

checker = None

@app.post("/predict")
async def predict_coverage(file: UploadFile, scenario: str = Form(...)):
    global checker
    try:
        # Use a unique temp file name to avoid conflicts
        pdf_path = f"temp_{int(time.time())}_{file.filename}"
        with open(pdf_path, 'wb') as f:
            f.write(await file.read())
        
        output_txt = "policy.txt"
        pdf_to_text.pdf_to_lines([pdf_path], output_txt)
        
        output_json = "embeddings.json"
        Embed_the_text_file.process_text_file(output_txt, output_json, model_name='all-MiniLM-L6-v2', batch_size=64)
        
        if not os.path.exists(output_json):
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")
        
        checker = FixedInsuranceCoverageChecker(embeddings_file=output_json)
        result = await checker.get_coverage_decision(scenario)
        result["sections"] = [{"text": text, "name": name, "relevance": sim} for text, name, sim in result["sections"]]
        
        clear_files.clear()
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return JSONResponse(content=result)
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he.detail)}")
        clear_files.clear()
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        clear_files.clear()
        if 'pdf_path' in locals() and os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    clear_files.clear()
    logger.info("API shutdown, files cleared")

if __name__ == "__main__":
    import uvicorn
    import time  # Added for unique filename
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)