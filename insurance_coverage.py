import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import aiohttp
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedInsuranceCoverageChecker:
    def __init__(self, embeddings_file: str = "embeddings.json", top_k: int = 3):
        """Initialize with proper file encoding and multi-clause retrieval"""
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.lines = [x['line'] for x in self.data]
            self.section_names = [x.get('section', f"Section_{i}") for i, x in enumerate(self.data)]
            self.embeddings = np.array([x['embedding'] for x in self.data])
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.top_k = top_k
            logger.info("Successfully loaded embeddings and initialized model")
        except UnicodeDecodeError:
            with open(embeddings_file, 'r', encoding='latin1') as f:
                self.data = json.load(f)
            self.lines = [x['line'] for x in self.data]
            self.section_names = [x.get('section', f"Section_{i}") for i, x in enumerate(self.data)]
            self.embeddings = np.array([x['embedding'] for x in self.data])
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embeddings with latin1 encoding fallback")
        except Exception as e:
            logger.error(f"Error loading files: {str(e)}")
            raise

    async def query_grok_api(self, sections: List[Tuple[str, str, float]], scenario: str) -> Tuple[str, str]:
        """Simulate querying Grok API with multiple sections for final decision"""
        logger.info("Querying Grok API with multiple sections")
        try:
            async with aiohttp.ClientSession() as session:
                # Simulated API response with weighted scoring
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
                            logger.info(f"Found positive keyword '{kw}' in {section_name}")
                    for kw, weight in exclusion_keywords.items():
                        if kw in section_lower:
                            score -= abs(weight) * sim
                            evidence.append(f"Negative: '{kw}' in {section_name} (Relevance: {sim:.1%})")
                            logger.info(f"Found negative keyword '{kw}' in {section_name}")

                # Scenario context influence
                scenario_lower = scenario.lower()
                for kw, weight in coverage_keywords.items():
                    if kw in scenario_lower:
                        score += weight * 0.5  # Lower weight for scenario keywords
                        evidence.append(f"Positive: '{kw}' in scenario")
                for kw, weight in exclusion_keywords.items():
                    if kw in scenario_lower:
                        score -= abs(weight) * 0.5
                        evidence.append(f"Negative: '{kw}' in scenario")

                decision = "Covered" if score > 0 else "Not Covered"
                explanation = "\n".join(evidence) or "No strong coverage indicators found"
                logger.info(f"Grok API decision: {decision}, Score: {score:.2f}")
                return decision, explanation
        except Exception as e:
            logger.error(f"Error querying Grok API: {str(e)}")
            return "Not Covered", f"API error: {str(e)}"

    async def get_coverage_decision(self, scenario: str) -> Dict:
        """Get coverage decision with multi-clause analysis and Grok API"""
        try:
            # Find top-k relevant policy lines
            scenario_embed = self.embedder.encode(scenario)
            sims = cosine_similarity([scenario_embed], self.embeddings)[0]
            top_indices = np.argsort(sims)[-self.top_k:][::-1]
            relevant_sections = [(self.lines[i], self.section_names[i], float(sims[i])) for i in top_indices]
            confidence = float(max(sims)) * 100  # Use max similarity as confidence

            # Local decision logic
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
                        logger.info(f"Local: Found positive keyword '{kw}' in {section_name}")
                for kw, weight in exclusion_keywords.items():
                    if kw in section_lower:
                        score -= abs(weight) * sim
                        evidence.append(f"Negative: '{kw}' in {section_name} (Relevance: {sim:.1%})")
                        logger.info(f"Local: Found negative keyword '{kw}' in {section_name}")

            if abs(score) > 1.5:  # Strong local decision
                decision = "Covered" if score > 0 else "Not Covered"
                explanation = "\n".join(evidence)
                source = "Local Decision"
            else:
                # Query Grok API for ambiguous cases
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

async def promptTheQuery():
    checker = FixedInsuranceCoverageChecker()
    
    print("Fixed Insurance Coverage Checker")
    print("Type 'exit' to quit\n")
    
    while True:
        scenario = input("Describe your scenario: ").strip()
        if scenario.lower() == 'exit':
            break
            
        result = await checker.get_coverage_decision(scenario)
        
        print("\n" + "="*60)
        print(f"SCENARIO: {scenario}")
        print(f"\nDECISION: {result['decision']} (Confidence: {result['confidence']:.1f}%)")
        print(f"\nEXPLANATION:\n{result['explanation']}")
        print("\nRELEVANT SECTIONS:")
        for i, (text, name, sim) in enumerate(result['sections'], 1):
            print(f"{i}. {name} [Relevance: {sim:.1f}%]: {text}")
        print(f"\nSOURCE: {result['source']}")
        print("="*60 + "\n")