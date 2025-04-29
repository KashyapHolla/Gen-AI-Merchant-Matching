from pathlib import Path
from langchain_ollama import ChatOllama

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPT_PATH = PROJECT_ROOT / "prompts"

# LLM Setup
parser_llm = ChatOllama(
    model="gemma3:1b",  # or "deepseek:latest"
    temperature=0.2
)

rerank_llm = ChatOllama(
    model="gemma3:12b",  # Using deepseek-r1:14b for reranking
    temperature=0.1,
    base_url="http://localhost:11434"
)

judge_llm = ChatOllama(
    model="gemma3:12b",  # Using gemma3:12b for judging
    temperature=0.0,
    base_url="http://localhost:11434"
)

# Utility Functions
def get_prompt(name: str) -> str:
    """Load a prompt from the prompts folder by filename (without .txt)."""
    prompt_file = PROMPT_PATH / f"{name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text()

# Elasticsearch credentials
ES_HOST = "http://localhost:9200"
ES_USERNAME = "elastic"
ES_PASSWORD = "your_password"  # Replace with your actual password

