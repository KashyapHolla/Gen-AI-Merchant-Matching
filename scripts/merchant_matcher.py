import asyncio
import json
import time
from typing import List, Dict, Tuple
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Import the hybrid search function
from hybrid_retrieval import search_merchant
# Import the transaction_parser from scripts.parser
from scripts.parser import transaction_parser
# Import all LLMs from config
from config import rerank_llm, judge_llm

# ------------------------------------------------------------------
# 1.  Parser helper
# ------------------------------------------------------------------
async def parse_descriptor(desc: str) -> Dict:
    """Parse the input descriptor string into a structured format"""
    
    # Use the existing parser function from scripts.parser
    result = transaction_parser(desc)
    
    # Convert merchant_id to integer if present
    if result.get("merchant_id"):
        try:
            result["merchant_id"] = int(result["merchant_id"])
        except ValueError:
            # If it's not a valid integer, keep as is
            pass
    
    return result

# ------------------------------------------------------------------
# 2.  Reranker helper for ONE candidate
# ------------------------------------------------------------------
def build_rerank_prompt(descriptor: str, candidate: Dict) -> List:
    """Build the prompt for reranking a candidate"""
    sys = SystemMessage(
        content=("You are a merchant matching expert. "
                 "Score 0-1 similarity between Descriptor and Candidate merchant. "
                 "A score of 1 means the descriptor perfectly matches the candidate merchant. "
                 "A score of 0 means they are completely different. "
                 "Consider brand name, location, and merchant ID if available. "
                 "Output JSON {\"score\":float, \"why\":string} with your reasoning.")
    )
    usr = HumanMessage(
        content=f"Descriptor: \"{descriptor}\"\nCandidate:\n{json.dumps(candidate)}")
    return [sys, usr]

def parse_rerank_output(text: str) -> Tuple[float, str]:
    """Parse the reranker output to extract score and reasoning"""
    try:
        # Try to parse the last line as JSON
        obj = json.loads(text.strip().splitlines()[-1])
        return obj["score"], obj["why"]
    except (json.JSONDecodeError, KeyError, IndexError):
        # If parsing fails, try to find a pattern in the text
        print(f"Warning: Failed to parse reranker output as JSON: {text}")
        
        # Default values
        score = 0.0
        why = "Failed to parse reasoning"
        
        # Look for score pattern
        import re
        score_pattern = r'"score"\s*:\s*([0-9.]+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass
                
        # Look for why pattern
        why_pattern = r'"why"\s*:\s*"([^"]+)"'
        why_match = re.search(why_pattern, text)
        if why_match:
            why = why_match.group(1)
            
        return score, why

# ------------------------------------------------------------------
# 3.  Optional judge helper
# ------------------------------------------------------------------
def build_judge_prompt(descriptor: str, candidate: Dict, rationale: str) -> List:
    """Build the prompt for judging a candidate"""
    sys = SystemMessage(
        content=("You are a judge evaluating merchant matches. "
                 "Determine if the rationale for matching contains hallucinations. "
                 "If the rationale references fields NOT present in either "
                 "descriptor or candidate, output {\"verdict\":\"reject\"}. "
                 "Otherwise output {\"verdict\":\"accept\"}. "
                 "Be strict about factual accuracy.")
    )
    usr = HumanMessage(
        content=f"Descriptor:\n{descriptor}\nCandidate:\n{json.dumps(candidate)}\n"
                f"Rationale:\n{rationale}")
    return [sys, usr]

def parse_judge_output(text: str) -> str:
    """Parse the judge output to extract the verdict"""
    try:
        # Try to parse the last line as JSON
        obj = json.loads(text.strip().splitlines()[-1])
        return obj["verdict"]
    except (json.JSONDecodeError, KeyError, IndexError):
        # If parsing fails, try to find a pattern in the text
        print(f"Warning: Failed to parse judge output as JSON: {text}")
        
        # Look for verdict pattern
        import re
        verdict_pattern = r'"verdict"\s*:\s*"([^"]+)"'
        verdict_match = re.search(verdict_pattern, text)
        if verdict_match:
            return verdict_match.group(1)
        
        # Default to accept if unable to parse
        return "accept"

# ------------------------------------------------------------------
# 4.  Public function
# ------------------------------------------------------------------
async def _async_match(descriptor: str,
                       top_k_search: int = 20,
                       judge: bool = True) -> Dict:
    """Async function to match a descriptor to the best merchant"""
    t0 = time.perf_counter()

    # ---- Stage 1: Parse the descriptor ---------------------------
    print(f"Parsing descriptor: {descriptor}")
    parsed = await parse_descriptor(descriptor)
    print(f"Parsed result: {json.dumps(parsed, indent=2)}")

    # ---- Stage 2: Retrieve candidates ---------------------------
    print(f"Retrieving candidates...")
    candidates = search_merchant(parsed)
    if not candidates:
        return {
            "error": "no_candidates", 
            "latency_ms": int((time.perf_counter()-t0)*1000),
            "parsed": parsed,
            "descriptor": descriptor
        }
    
    print(f"Retrieved {len(candidates)} candidates")

    # ---- Stage 3: Rerank candidates -----------------------------
    print(f"Reranking candidates...")
    prompts = [build_rerank_prompt(descriptor, c) for c in candidates[:top_k_search]]
    rerank_resp = await rerank_llm.agenerate(prompts)
    scored = []
    
    for i, (cand, gen) in enumerate(zip(candidates[:top_k_search], rerank_resp.generations)):
        score, why = parse_rerank_output(gen[0].text)
        print(f"Candidate {i+1} score: {score}")
        cand.update({"score": score, "why": why})
        scored.append(cand)
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    best = scored[0]
    print(f"Best candidate: {json.dumps(best, indent=2)}")

    # ---- Stage 4: Judge the match (optional) --------------------
    if judge and len(scored) > 1:
        print(f"Judging the best match...")
        judge_prompt = build_judge_prompt(descriptor, best, best["why"])
        judge_resp = await judge_llm.agenerate([judge_prompt])
        verdict_txt = judge_resp.generations[0][0].text
        verdict = parse_judge_output(verdict_txt)
        
        if verdict == "reject":
            print(f"Top match rejected by judge. Using second-best match.")
            best = scored[1]
            best["why"] += " | Note: top-1 rejected by judge for hallucinations."

    lat_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "match": best,
        "latency_ms": lat_ms,
        "parsed": parsed,
        "descriptor": descriptor
    }

def match_descriptor(descriptor: str, **kwargs) -> Dict:
    """Synchronous wrapper for the async matching function"""
    return asyncio.run(_async_match(descriptor, **kwargs))

if __name__ == "__main__":
    # Example usage
    test_descriptors = [
        "VRIERTRS #727612092139916032 Monterey Park CA",
        "NLDOJXLQ #245178307025547008 La Verne CA",
        "DGAVXRSA #97032797689821696 Phoenix AZ"
    ]
    
    for desc in test_descriptors:
        print(f"\n=============== Testing: {desc} ===============")
        result = match_descriptor(desc, top_k_search=10, judge=True)
        
        print(f"Result latency: {result['latency_ms']} ms")
        print(f"Parsed as: {json.dumps(result['parsed'], indent=2)}")
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            match = result["match"]
            print(f"Best match:\n- Brand: {match['brand']}")
            print(f"- City: {match['city']}")
            print(f"- State: {match['state']}")
            print(f"- ID: {match['merchant_id']}")
            print(f"- Score: {match['score']}")
            print(f"- Reasoning: {match['why']}")
        
        print("=" * 50) 