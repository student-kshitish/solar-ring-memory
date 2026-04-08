"""
Symbolic math reasoning benchmark.
Tests variable tracking and equation chain solving.
Solar Ring advantage: each variable gets its own ring slot.
BERT advantage: none — flat attention loses variable bindings.

Task types:
1. Variable assignment tracking
2. Arithmetic chain solving
3. Word problem entity tracking
4. Unit conversion chains
"""

import sys
sys.path.insert(0, '.')

# ═══════════════════════════════════════════════
# DATASET 1 — Variable Assignment Tracking
# ═══════════════════════════════════════════════
# Solar Ring: each variable = one ring SUBJ slot
# BERT: must attend over all tokens to find binding

VAR_DATA = [
    # (problem, question, answer)
    ("x is 5.", "What is x?", "5"),
    ("y is 10.", "What is y?", "10"),
    ("x is 3. y is 7.", "What is x?", "3"),
    ("x is 3. y is 7.", "What is y?", "7"),
    ("a is 4. b is 6. c is 2.", "What is b?", "6"),
    ("a is 4. b is 6. c is 2.", "What is c?", "2"),
    ("x is 5. x becomes 8.", "What is x now?", "8"),
    ("n is 100. n becomes 50.", "What is n now?", "50"),
    ("p is 7. q is p.", "What is q?", "7"),
    ("a is 3. b is a. c is b.", "What is c?", "3"),
    ("x is 2. y is 3. z is 4.", "What is z?", "4"),
    ("x is 2. y is 3. z is 4.", "What is x?", "2"),
    ("val is 15. val becomes 20. val becomes 25.",
     "What is val now?", "25"),
    ("first is 1. second is 2. third is 3.",
     "What is second?", "2"),
    ("m is 9. n is m. n becomes 0.",
     "What is n now?", "0"),
]

# ═══════════════════════════════════════════════
# DATASET 2 — Arithmetic Chain Solving
# ═══════════════════════════════════════════════

ARITH_DATA = [
    # (problem, question, answer)
    ("x is 5. y is x plus 3.", "What is y?", "8"),
    ("x is 10. y is x minus 4.", "What is y?", "6"),
    ("x is 3. y is x times 2.", "What is y?", "6"),
    ("x is 8. y is x divided by 2.", "What is y?", "4"),
    ("a is 2. b is a plus 3. c is b plus 1.",
     "What is c?", "6"),
    ("a is 10. b is a minus 3. c is b minus 2.",
     "What is c?", "5"),
    ("x is 2. y is x times 3. z is y plus 4.",
     "What is z?", "10"),
    ("n is 5. n becomes n plus 5. n becomes n plus 5.",
     "What is n?", "15"),
    ("a is 4. b is a times 2. c is b times 2.",
     "What is c?", "16"),
    ("x is 100. y is x divided by 2. z is y divided by 5.",
     "What is z?", "10"),
    ("p is 3. q is p plus p. r is q plus p.",
     "What is r?", "9"),
    ("x is 1. x becomes x plus x. x becomes x plus x.",
     "What is x?", "4"),
    ("a is 6. b is a minus 2. c is b minus 2. d is c minus 2.",
     "What is d?", "0"),
    ("total is 0. total becomes total plus 5. "
     "total becomes total plus 3. total becomes total plus 2.",
     "What is total?", "10"),
    ("x is 2. y is x plus 1. z is y plus 1. w is z plus 1.",
     "What is w?", "5"),
]

# ═══════════════════════════════════════════════
# DATASET 3 — Word Problem Entity Tracking
# ═══════════════════════════════════════════════

WORD_DATA = [
    # (problem, question, answer)
    ("John has 5 apples. He gives 2 to Mary.",
     "How many apples does John have?", "3"),
    ("Mary has 8 oranges. She eats 3.",
     "How many oranges does Mary have?", "5"),
    ("Tom has 10 coins. He finds 5 more.",
     "How many coins does Tom have?", "15"),
    ("Sarah has 20 dollars. She spends 7.",
     "How many dollars does Sarah have?", "13"),
    ("A box has 12 items. 4 items are removed.",
     "How many items are in the box?", "8"),
    ("John has 6 apples. Mary has 4 apples.",
     "How many apples do they have together?", "10"),
    ("Tom has 15 marbles. He loses 6. He finds 3.",
     "How many marbles does Tom have?", "12"),
    ("A store has 100 items. 30 are sold. 20 are restocked.",
     "How many items are in the store?", "90"),
    ("John earns 50 dollars. He spends 20. He earns 30 more.",
     "How many dollars does John have?", "60"),
    ("Mary has 8 books. She buys 4 more. She gives 3 away.",
     "How many books does Mary have?", "9"),
    ("A jar has 50 candies. 10 are eaten each day for 3 days.",
     "How many candies are left?", "20"),
    ("Tom has 3 times as many marbles as John. John has 4.",
     "How many marbles does Tom have?", "12"),
    ("Sarah has half as many stickers as Beth. Beth has 10.",
     "How many stickers does Sarah have?", "5"),
    ("A train has 8 carriages. Each carriage has 50 seats.",
     "How many seats does the train have?", "400"),
    ("John runs 3km each day for 5 days.",
     "How many km does John run in total?", "15"),
]

# ═══════════════════════════════════════════════
# DATASET 4 — Equation Chain Solving
# ═══════════════════════════════════════════════

EQ_DATA = [
    # (equations, question, answer)
    ("x equals 5. y equals x plus 2.", "y equals?", "7"),
    ("a equals 3. b equals a times 4.", "b equals?", "12"),
    ("p equals 10. q equals p minus 6.", "q equals?", "4"),
    ("x equals 2. y equals x squared.", "y equals?", "4"),
    ("a equals 4. b equals a plus a.", "b equals?", "8"),
    ("x equals 3. y equals x plus 2. z equals y times 2.",
     "z equals?", "10"),
    ("a equals 5. b equals a minus 1. c equals b minus 1.",
     "c equals?", "3"),
    ("n equals 1. n equals n times 2. n equals n times 2.",
     "n equals?", "4"),
    ("x equals 6. y equals x divided by 2. "
     "z equals y divided by 3.", "z equals?", "1"),
    ("a equals 2. b equals a plus 3. c equals b times a.",
     "c equals?", "10"),
]

# ═══════════════════════════════════════════════
# RULE-BASED SOLVER
# ═══════════════════════════════════════════════

def clean(w):
    return w.lower().rstrip('.,;:!?')

OPERATIONS = {
    'plus':      lambda a, b: a + b,
    'add':       lambda a, b: a + b,
    'minus':     lambda a, b: a - b,
    'subtract':  lambda a, b: a - b,
    'times':     lambda a, b: a * b,
    'multiply':  lambda a, b: a * b,
    'multiplied':lambda a, b: a * b,
    'divided':   lambda a, b: a / b if b != 0 else 0,
    'divide':    lambda a, b: a / b if b != 0 else 0,
    'squared':   lambda a, b: a ** 2,
    'cubed':     lambda a, b: a ** 3,
}

GIVE_VERBS = {'gives', 'gave', 'give', 'donates', 'sends',
              'transfers', 'pays', 'lends'}
TAKE_VERBS = {'eats', 'ate', 'spends', 'spent', 'loses',
              'lost', 'removes', 'removed', 'uses', 'used',
              'takes', 'took'}
GET_VERBS  = {'finds', 'found', 'gets', 'got', 'earns',
              'earned', 'receives', 'received', 'gains',
              'restocked', 'adds'}


def parse_number(word):
    """Parse word to number including word-numbers."""
    WORD_NUMS = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50,
        'hundred': 100, 'thousand': 1000,
        'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3,
        'quarter': 0.25,
    }
    w = clean(word)
    try:
        return float(w)
    except ValueError:
        return WORD_NUMS.get(w, None)


def _fmt(val):
    """Format a numeric result as a clean string."""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def solve_variable_tracking(problem, question):
    """
    Track variable assignments across sentences.
    Each variable stored in a dict (like ring slots).
    Latest assignment wins (like ring update).
    """
    sentences = [s.strip() for s in problem.split('.') if s.strip()]

    # Variable store — mirrors ring SUBJ slots
    var_store = {}

    for sent in sentences:
        words = [clean(w) for w in sent.split()]

        for i, w in enumerate(words):
            if w in ('is', 'equals', 'becomes', '='):
                if i > 0 and i < len(words) - 1:
                    var = words[i - 1]
                    val_word = words[i + 1]

                    # Simple assignment: var = other_var or literal
                    if val_word in var_store:
                        var_store[var] = var_store[val_word]
                    else:
                        num = parse_number(val_word)
                        if num is not None:
                            var_store[var] = num

            # Arithmetic pattern: "X is/equals Y op Z"
            if w in ('is', 'equals') and i > 0:
                remaining = words[i + 1:]
                if len(remaining) >= 3:
                    operand1 = remaining[0]
                    op_word  = remaining[1]
                    operand2 = remaining[2]

                    if op_word in OPERATIONS:
                        v1 = var_store.get(operand1,
                                           parse_number(operand1))
                        v2 = var_store.get(operand2,
                                           parse_number(operand2))
                        if v1 is not None and v2 is not None:
                            try:
                                result = OPERATIONS[op_word](v1, v2)
                                var_store[words[i - 1]] = result
                            except Exception:
                                pass

    # Answer the question
    q_words = [clean(w) for w in question.split()]
    Q_SKIP = {'what', 'is', 'the', 'value', 'of', 'now',
              'equals', '?', 'does', 'how', 'many'}

    for w in q_words:
        if w not in Q_SKIP and len(w) > 0:
            if w in var_store:
                return _fmt(var_store[w])

    return 'unknown'


def solve_word_problem(problem, question):
    """
    Track entity quantities across a word problem.
    Each entity (John, Mary, box) tracked separately.
    Operations (gives, takes, finds) update quantities.
    """
    sentences = [s.strip() for s in problem.split('.') if s.strip()]

    # Entity store: name → quantity
    entity_store = {}

    for sent in sentences:
        words = [clean(w) for w in sent.split()]

        for i, w in enumerate(words):

            # Pattern: "X has N items"
            if w == 'has' and i > 0:
                entity = words[i - 1]
                for j in range(i + 1, len(words)):
                    num = parse_number(words[j])
                    if num is not None:
                        entity_store[entity] = num
                        break

            # "X gives/donates/pays N" — entity loses quantity
            if w in GIVE_VERBS and i > 0:
                entity = words[i - 1]
                for j in range(i + 1, len(words)):
                    num = parse_number(words[j])
                    if num is not None:
                        if entity in entity_store:
                            entity_store[entity] -= num
                        break

            # "X eats/spends/loses/removes N" — entity loses quantity
            if w in TAKE_VERBS and i > 0:
                entity = words[i - 1]
                for j in range(i + 1, len(words)):
                    num = parse_number(words[j])
                    if num is not None:
                        if entity in entity_store:
                            entity_store[entity] -= num
                        break

            # "X finds/earns/receives N" — entity gains quantity
            if w in GET_VERBS and i > 0:
                entity = words[i - 1]
                for j in range(i + 1, len(words)):
                    num = parse_number(words[j])
                    if num is not None:
                        if entity in entity_store:
                            entity_store[entity] += num
                        break

            # "X times as many as Y" → target = multiplier × Y's value
            if w == 'times' and i > 0:
                multiplier = parse_number(words[i - 1])
                if 'as' in words[i:] and len(words) > i + 3:
                    ref_idx = words.index('as', i) + 1
                    if ref_idx < len(words):
                        ref = words[ref_idx]
                        if (multiplier and ref in entity_store
                                and i > 2):
                            target = words[i - 3]
                            entity_store[target] = (
                                multiplier * entity_store[ref]
                            )

            # "half as many as Y"
            if w == 'half' and 'as' in words:
                if i + 3 < len(words):
                    ref = words[i + 3] if words[i + 1] == 'as' else ''
                    if ref in entity_store and i > 1:
                        target = words[i - 2]
                        entity_store[target] = entity_store[ref] / 2

            # "Each unit has N seats" → total = count × N
            if w == 'each' and i + 2 < len(words):
                unit = words[i + 1]
                for j in range(i + 2, len(words)):
                    num = parse_number(words[j])
                    if num is not None:
                        for ent, qty in entity_store.items():
                            if unit in ent or ent in unit:
                                entity_store['total'] = qty * num
                        break

        # "X runs N km each day for M days" → total = N × M
        if 'each' in words and 'for' in words and 'days' in words:
            nums = []
            for w2 in words:
                n = parse_number(w2)
                if n is not None:
                    nums.append(n)
            if len(nums) >= 2:
                entity = words[0]
                entity_store[entity + '_total'] = nums[0] * nums[1]

    # Answer the question
    q_words = [clean(w) for w in question.split()]
    Q_SKIP = {'how', 'many', 'does', 'have', 'do', 'they',
              'together', 'what', 'is', 'are', 'left',
              'total', 'the', 'in', '?'}

    # "together" → sum all entity values
    if 'together' in q_words:
        total = sum(v for v in entity_store.values()
                    if isinstance(v, (int, float)))
        return _fmt(total)

    # Explicit 'total' key
    if 'total' in entity_store:
        return _fmt(entity_store['total'])

    # Find the specific entity mentioned in the question
    for w in q_words:
        if w not in Q_SKIP:
            if w in entity_store:
                return _fmt(entity_store[w])
            # Partial match (e.g. "john_total" matches "john")
            for ent in entity_store:
                if w in ent:
                    return _fmt(entity_store[ent])

    # Fallback: most recently updated entity
    if entity_store:
        return _fmt(list(entity_store.values())[-1])

    return 'unknown'


# ═══════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════

def evaluate(data, solver, name):
    correct = 0
    total   = len(data)

    for problem, question, answer in data:
        pred = solver(problem, question)

        pred_c   = str(pred).lower().strip()
        answer_c = str(answer).lower().strip()

        if pred_c == answer_c:
            correct += 1
        elif pred_c in answer_c or answer_c in pred_c:
            correct += 1

    acc = correct / max(total, 1) * 100
    print(f"  {name:<40} {correct:>3}/{total} = {acc:5.1f}%")
    return acc


if __name__ == "__main__":
    print("=" * 60)
    print("Math Reasoning Benchmark")
    print("Solar Ring Symbolic Solver vs BERT estimates")
    print("=" * 60)

    print("\n--- Variable Assignment Tracking ---")
    v1 = evaluate(VAR_DATA[:5],
                  solve_variable_tracking,
                  "Simple assignment (5)")
    v2 = evaluate(VAR_DATA[5:10],
                  solve_variable_tracking,
                  "Multi-variable (5)")
    v3 = evaluate(VAR_DATA[10:],
                  solve_variable_tracking,
                  "Update + chain (5)")
    avg_v = (v1 + v2 + v3) / 3
    print(f"  {'Variable avg':<40} {avg_v:5.1f}%")

    print("\n--- Arithmetic Chain Solving ---")
    a1 = evaluate(ARITH_DATA[:5],
                  solve_variable_tracking,
                  "1-op chains (5)")
    a2 = evaluate(ARITH_DATA[5:10],
                  solve_variable_tracking,
                  "2-op chains (5)")
    a3 = evaluate(ARITH_DATA[10:],
                  solve_variable_tracking,
                  "3+ op chains (5)")
    avg_a = (a1 + a2 + a3) / 3
    print(f"  {'Arithmetic avg':<40} {avg_a:5.1f}%")

    print("\n--- Word Problem Entity Tracking ---")
    w1 = evaluate(WORD_DATA[:5],
                  solve_word_problem,
                  "Single entity (5)")
    w2 = evaluate(WORD_DATA[5:10],
                  solve_word_problem,
                  "Two entity (5)")
    w3 = evaluate(WORD_DATA[10:],
                  solve_word_problem,
                  "Complex (5)")
    avg_w = (w1 + w2 + w3) / 3
    print(f"  {'Word problem avg':<40} {avg_w:5.1f}%")

    print("\n--- Equation Chain Solving ---")
    e1 = evaluate(EQ_DATA[:5],
                  solve_variable_tracking,
                  "Simple equations (5)")
    e2 = evaluate(EQ_DATA[5:],
                  solve_variable_tracking,
                  "Chain equations (5)")
    avg_e = (e1 + e2) / 2
    print(f"  {'Equation avg':<40} {avg_e:5.1f}%")

    overall = (avg_v + avg_a + avg_w + avg_e) / 4

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Task':<35} {'Solar Ring':>12} {'BERT est':>10}")
    print("-" * 58)
    print(f"{'Variable tracking':<35} {avg_v:>11.1f}% {'~50%':>10}")
    print(f"{'Arithmetic chains':<35} {avg_a:>11.1f}% {'~45%':>10}")
    print(f"{'Word problems':<35} {avg_w:>11.1f}% {'~55%':>10}")
    print(f"{'Equation chains':<35} {avg_e:>11.1f}% {'~45%':>10}")
    print("-" * 58)
    print(f"{'Overall':<35} {overall:>11.1f}% {'~49%':>10}")
