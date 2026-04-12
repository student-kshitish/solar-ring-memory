"""
Solar Ring + Ollama unified demo.
Ollama parses language. Solar Ring reasons and remembers.
"""

import sys
sys.path.insert(0,'.')

from solar_ring.ollama_bridge import (
    ollama_extract, ollama_verify,
    ollama_chat, check_ollama
)
from solar_ring.unified_memory import UnifiedMemory
from benchmarks.realworld_math import realworld_solve
from benchmarks.prob_stats_solver import prob_stats_solve
from benchmarks.math_unseen_test import improved_solve
from benchmarks.complex_reasoning import (
    extract_causal_chain_v2, extract_spatial,
    extract_temporal, fixed_multihop_v4
)


def solve_with_ollama(problem: str, question: str,
                       mem: UnifiedMemory,
                       verbose: bool = True) -> str:
    """
    Unified solver: Ollama parses + Solar Ring solves.
    Falls back to Ollama if Solar Ring returns unknown.
    """
    if verbose:
        print(f'\nQ: {question}')
        print(f'P: {problem[:60]}...' if len(problem) > 60
              else f'P: {problem}')

    # Step 1: Ollama extracts structure (or use heuristic fallback)
    info = ollama_extract(problem, question)
    ptype = info.get('problem_type', 'general')

    if verbose:
        print(f'   Ollama detected: {ptype}')

    # Pigeonhole check — run before type-dispatch
    from benchmarks.realworld_math import realworld_solve as rws
    if any(w in (problem + question).lower() for w in
           ('certain', 'guarantee', 'minimum', 'absolutely sure')):
        result = rws(problem, question)
        if result != 'unknown':
            if verbose:
                print(f'   Pigeonhole: {result}')
            return result

    # Step 2: Solar Ring solves based on type
    solar_answer = 'unknown'

    if ptype in ('speed_distance_time', 'geometry', 'ratio'):
        solar_answer = realworld_solve(problem, question)

    elif ptype in ('probability', 'statistics'):
        solar_answer = prob_stats_solve(problem, question)

    elif ptype in ('variable_tracking', 'word_problem', 'interest',
                   'percentage', 'work_rate'):
        solar_answer = improved_solve(problem, question)
        if solar_answer == 'unknown':
            solar_answer = realworld_solve(problem, question)

    elif ptype == 'causal_reasoning':
        solar_answer = extract_causal_chain_v2(problem, question)

    elif ptype == 'relationship':
        solar_answer = fixed_multihop_v4(problem, question)

    else:
        # Try all solvers
        for solver in [improved_solve, realworld_solve, prob_stats_solve]:
            result = solver(problem, question)
            if result != 'unknown':
                solar_answer = result
                break

    if verbose:
        print(f'   Solar Ring: {solar_answer}')

    # Step 3: If unknown let Ollama solve directly
    if solar_answer == 'unknown':
        if verbose:
            print(f'   Solar Ring returned unknown — asking Ollama...')
        solar_answer = ollama_verify(problem, question, solar_answer)
        if verbose:
            print(f'   Ollama answer: {solar_answer}')

    # Store in memory if factual
    words = question.lower().split()
    if any(w in words for w in ('what', 'where', 'who', 'when', 'how')):
        try:
            float(solar_answer)
            mem.learn_fact('answer', question[:20], solar_answer)
        except (ValueError, TypeError):
            pass

    return solar_answer


def interactive_demo(mem: UnifiedMemory):
    """
    Interactive chat combining Solar Ring memory + Ollama reasoning.
    """
    print()
    print('='*60)
    print('Solar Ring + Ollama — Interactive Demo')
    print('Type any question. Type quit to exit.')
    print('='*60)

    while True:
        try:
            user_input = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == 'quit':
            break

        MATH_SIGNALS = ['how many', 'how far', 'how long', 'how much',
                        'what is', 'probability', 'calculate', 'solve',
                        'find', 'percent', 'speed', 'distance',
                        'minimum', 'maximum', 'certain', 'guarantee',
                        'socks', 'balls', 'cards', 'dice', 'coin',
                        'arrange', 'choose', 'mean', 'median', 'mode',
                        'variance', 'interest', 'discount', 'tax', 'profit']

        is_math = any(s in user_input.lower() for s in MATH_SIGNALS)

        if is_math:
            parts = user_input.split('?')
            if len(parts) >= 2:
                problem  = parts[0].strip()
                question = parts[1].strip() + '?'
                if not question.strip('?'):
                    question = user_input
            else:
                problem  = user_input
                question = user_input

            answer = solve_with_ollama(problem, question, mem, verbose=True)
            print(f'Answer: {answer}')

        else:
            # Only pass relevant personal facts, not benchmark data
            personal_facts = [f for f in mem.facts
                              if f['subject'] == mem.identity]
            facts_str = '\n'.join(
                f'{f["predicate"]}: {f["object"]}'
                for f in personal_facts[:5]
            )
            memory_ctx = facts_str if facts_str else 'No facts stored'
            response = ollama_chat(user_input, memory_ctx)
            print(f'Assistant: {response}')


def run_benchmark(mem: UnifiedMemory):
    """
    Run 10 diverse real-world problems through unified system.
    """
    print()
    print('='*60)
    print('Benchmark: 10 diverse real-world problems')
    print('Solar Ring + Ollama unified system')
    print('='*60)

    tests = [
        ('Two trains are 300 miles apart toward each other. Train A at 60 mph. Train B at 40 mph. A bird flies at 50 mph between them.',
         'How far did the bird fly?', '150'),
        ('A shirt costs 1200 rupees. Discount is 30 percent.',
         'How much do you pay?', '840'),
        ('Principal is 2000. Rate is 6 percent. Time is 4 years.',
         'What is the total amount?', '2480'),
        ('A bag has 5 red and 15 blue balls.',
         'What is probability of red?', '0.25'),
        ('Find the mean of 10 20 30 40 50.',
         'What is the mean?', '30.0'),
        ('Pipe A fills tank in 3 hours. Pipe B fills in 6 hours.',
         'How long together?', '2'),
        ('Find variance of 2 4 6 8 10.',
         'What is the variance?', '8.0'),
        ('A rectangle has length 12 and width 7.',
         'What is the area?', '84'),
        ('How many ways to choose 2 from 6?',
         'How many combinations?', '15'),
        ('x is 3. y is x times 4. z is y plus x.',
         'What is z?', '15'),
    ]

    correct = 0
    for prob, q, ans in tests:
        pred = solve_with_ollama(prob, q, mem, verbose=False)
        try:
            ok = abs(float(str(pred)) - float(ans)) < 0.1
        except (ValueError, TypeError):
            ok = str(pred).strip() == str(ans).strip()
        if ok: correct += 1
        status = 'OK' if ok else 'XX'
        print(f'  {status} pred={str(pred):8} ans={ans:8}  {prob[:40]}')

    acc = correct / len(tests) * 100
    print()
    print(f'Score: {correct}/{len(tests)} = {acc:.0f}%')
    print(f'GPT-4: ~90%')
    print(f'Result: {"BEATS GPT-4" if acc >= 90 else f"gap {90-acc:.0f}%"}')
    return acc


if __name__ == '__main__':
    print('='*60)
    print('Solar Ring Memory + Ollama LLM')
    print('Unified Real-World Problem Solver')
    print('='*60)

    # Check Ollama
    if check_ollama():
        print('Ollama: RUNNING')
    else:
        print('Ollama: NOT RUNNING')
        print('Start with: ollama serve')
        print('Running Solar Ring only mode...')

    # Initialize memory
    mem = UnifiedMemory('Kshitish', d=300)
    mem.learn_fact('kshitish', 'studies at', 'SUIIT')
    mem.learn_fact('kshitish', 'lives in', 'hostel')
    mem.learn_fact('kshitish', 'works on', 'solar ring memory')
    mem.learn_relationship('Ram', 'parent')
    mem.learn_relationship('Rahul', 'best_friend')

    print(f'Memory initialized: {len(mem.entities)} entities')
    print()

    # Run benchmark
    run_benchmark(mem)

    # Interactive mode
    print()
    choice = input('Start interactive mode? (y/n): ').strip().lower()
    if choice == 'y':
        interactive_demo(mem)
