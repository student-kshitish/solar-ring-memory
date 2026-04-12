"""
Bridge between Solar Ring Memory and Ollama LLM.
Ollama parses natural language.
Solar Ring handles reasoning and memory.
Together they solve arbitrary real-world problems.
"""

import json
import requests
import sys
sys.path.insert(0,'.')

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'llama3.2:3b'

def ollama_extract(problem: str, question: str) -> dict:
    """
    Use Ollama to extract structured info from problem.
    Returns dict with problem_type, numbers, entities, formula.
    """
    prompt = f"""Extract information from this math/reasoning problem.
Return ONLY valid JSON, no other text.

Problem: {problem}
Question: {question}

Return JSON with these fields:
{{
  "problem_type": one of [speed_distance_time, percentage, interest, work_rate, probability, statistics, geometry, ratio, word_problem, variable_tracking, causal_reasoning, relationship, general],
  "numbers": [list of all numbers found],
  "entities": [list of named things like people, objects],
  "keywords": [important words that indicate the formula],
  "formula_hint": brief description of what formula to use,
  "direction": "toward" or "away" or "same" or "none"
}}

JSON only:"""

    try:
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1}
        }, timeout=30)

        if response.status_code == 200:
            text = response.json().get('response', '')
            # Extract JSON from response
            start = text.find('{')
            end   = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
    except Exception:
        pass

    return {
        'problem_type': 'general',
        'numbers': [],
        'entities': [],
        'keywords': [],
        'formula_hint': '',
        'direction': 'none'
    }


def ollama_verify(problem: str, question: str,
                   solar_answer: str) -> str:
    """
    Use Ollama to verify and explain Solar Ring answer.
    If Solar Ring returned unknown Ollama tries to solve directly.
    """
    prompt = f"""Problem: {problem}
Question: {question}
Solar Ring computed: {solar_answer}

If Solar Ring answer is 'unknown' or seems wrong solve it yourself.
Give ONLY the final numerical answer as a single number or short phrase.
No explanation. Just the answer:"""

    try:
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.1}
        }, timeout=30)

        if response.status_code == 200:
            text = response.json().get('response', '').strip()
            # Extract first number or short answer
            for w in text.split():
                w_clean = w.rstrip('.,;')
                try:
                    float(w_clean)
                    return w_clean
                except ValueError:
                    continue
            return text[:50]
    except Exception:
        pass

    return solar_answer


def ollama_chat(message: str,
                memory_context: str = '') -> str:
    """
    General chat with Ollama enhanced by Solar Ring memory.
    memory_context = facts Solar Ring has stored about the user.
    """
    system = f"""You are a helpful assistant with access to structured memory.
Known facts about the user:
{memory_context if memory_context else 'No facts stored yet.'}

Answer concisely and accurately."""

    prompt = f"{system}\n\nUser: {message}\nAssistant:"

    try:
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': 0.7, 'max_tokens': 200}
        }, timeout=30)

        if response.status_code == 200:
            return response.json().get('response', '').strip()
    except Exception as e:
        return f'Ollama not available: {e}'

    return 'No response'


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        return r.status_code == 200
    except Exception:
        return False
