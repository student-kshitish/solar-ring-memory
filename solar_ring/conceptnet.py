"""
ConceptNet knowledge injection for Winograd resolution.
Queries local ConceptNet facts without API calls.
Hardcoded relations for common Winograd objects.
"""

CONCEPTNET = {
    # Physical size relations
    'trophy':     {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': False},
    'suitcase':   {'size': 'large',  'weight': 'medium', 'fragile': False, 'animate': False},
    'ball':       {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': False},
    'window':     {'size': 'large',  'weight': 'medium', 'fragile': True,  'animate': False},
    'cup':        {'size': 'small',  'weight': 'light',  'fragile': True,  'animate': False},
    'plate':      {'size': 'medium', 'weight': 'light',  'fragile': True,  'animate': False},
    'car':        {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': False},
    'tree':       {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': False},
    'rock':       {'size': 'medium', 'weight': 'heavy',  'fragile': False, 'animate': False},
    'bottle':     {'size': 'small',  'weight': 'light',  'fragile': True,  'animate': False},
    'vase':       {'size': 'medium', 'weight': 'light',  'fragile': True,  'animate': False},
    'table':      {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': False},
    'box':        {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': False},
    'bag':        {'size': 'medium', 'weight': 'light',  'fragile': False, 'animate': False},
    'chicken':    {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': True},
    'sandwich':   {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': False},
    'pitcher':    {'size': 'medium', 'weight': 'medium', 'fragile': True,  'animate': False},
    'atom':       {'size': 'tiny',   'weight': 'tiny',   'fragile': False, 'animate': False},
    'coin':       {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': False},
    'computer':   {'size': 'medium', 'weight': 'medium', 'fragile': True,  'animate': False},
    'program':    {'size': None,     'weight': None,     'fragile': False, 'animate': False},
    'fish':       {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': True},
    'bird':       {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': True},
    'cat':        {'size': 'small',  'weight': 'light',  'fragile': False, 'animate': True},
    'dog':        {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'elephant':   {'size': 'huge',   'weight': 'huge',   'fragile': False, 'animate': True},
    'mouse':      {'size': 'tiny',   'weight': 'tiny',   'fragile': False, 'animate': True},
    'man':        {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': True},
    'woman':      {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'boy':        {'size': 'medium', 'weight': 'light',  'fragile': False, 'animate': True},
    'girl':       {'size': 'medium', 'weight': 'light',  'fragile': False, 'animate': True},
    'baby':       {'size': 'small',  'weight': 'light',  'fragile': True,  'animate': True},
    'athlete':    {'size': 'large',  'weight': 'heavy',  'fragile': False, 'animate': True},
    'coach':      {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'scientist':  {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'philosopher':{'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'lawyer':     {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
    'criminal':   {'size': 'medium', 'weight': 'medium', 'fragile': False, 'animate': True},
}

CONTEXT_ADJECTIVES = {
    'big':     ('size',    ['large', 'huge']),
    'large':   ('size',    ['large', 'huge']),
    'huge':    ('size',    ['huge']),
    'small':   ('size',    ['small', 'tiny']),
    'tiny':    ('size',    ['tiny']),
    'little':  ('size',    ['small', 'tiny']),
    'heavy':   ('weight',  ['heavy', 'huge']),
    'light':   ('weight',  ['light', 'tiny']),
    'fragile': ('fragile', [True]),
    'strong':  ('fragile', [False]),
    'weak':    ('weight',  ['tiny', 'light']),
    'fast':    ('speed',   ['fast']),
    'slow':    ('speed',   ['slow']),
}


def get_properties(word: str) -> dict:
    return CONCEPTNET.get(word.lower(), {})


def conceptnet_score(pronoun_text: str, candidate_text: str,
                     context_sentence: str) -> float:
    """
    Score how likely candidate is the pronoun referent
    using ConceptNet knowledge.

    Returns score adjustment: positive = more likely
                              negative = less likely
    """
    words     = context_sentence.lower().split()
    candidate = candidate_text.lower().strip('.,')
    pronoun   = pronoun_text.lower().strip()

    props = get_properties(candidate)
    if not props:
        return 0.0

    score = 0.0

    # Strip punctuation from all words for matching
    words_clean = [w.rstrip('.,!?;:') for w in words]

    # Rule 1: animate/inanimate matches pronoun
    is_animate = props.get('animate', None)
    if pronoun in ('he', 'him', 'his', 'she', 'her', 'hers',
                   'they', 'them', 'their'):
        if is_animate is True:
            score += 0.5
        elif is_animate is False:
            score -= 0.5

    if pronoun in ('it', 'its', 'which', 'that'):
        if is_animate is False:
            score += 0.5
        elif is_animate is True:
            score -= 0.5

    # Rule 2: context adjectives matched against candidate properties
    for word in words_clean:
        if word not in CONTEXT_ADJECTIVES:
            continue
        prop_name, matching_values = CONTEXT_ADJECTIVES[word]
        candidate_prop = props.get(prop_name)
        if candidate_prop in matching_values:
            score += 0.8   # strong match
        elif candidate_prop is not None:
            score -= 0.4   # mismatch

    # Rule 3: causal verbs — only directional rules to avoid symmetric harm
    # "too big to fit" — the thing that is too big is the larger object
    # (only boost if it's notably large; neutral otherwise)
    if 'fit' in words_clean or 'fitted' in words_clean:
        if props.get('size') in ('large', 'huge'):
            score += 0.3

    # "lifted/carried" — heavy objects are harder to lift
    if 'lifted' in words_clean or 'carried' in words_clean:
        if props.get('weight') in ('heavy', 'huge'):
            score += 0.4

    return score


_ARTICLES = {'a', 'an', 'the', 'this', 'that', 'these', 'those'}


def _extract_head_noun(text: str) -> str:
    """
    Extract the first content word from a candidate string,
    skipping leading articles (a/an/the/this/that).
    e.g. "The trophy was too big." → "trophy"
         "Lisa had given help."    → "lisa"
    """
    for word in text.split():
        clean = word.lower().rstrip('.,!?')
        if clean and clean not in _ARTICLES:
            return clean
    return text.split()[0].lower().rstrip('.,')


def apply_conceptnet_to_winograd(sentence: str,
                                  correct: str,
                                  wrong: str) -> tuple:
    """
    Apply ConceptNet to score correct vs wrong candidate.
    Returns (score_correct_adj, score_wrong_adj)
    """
    words = sentence.lower().split()

    PRONOUNS = {'it', 'its', 'he', 'she', 'they', 'him', 'her',
                'them', 'who', 'which', 'that'}
    pronoun = None
    for w in words:
        if w in PRONOUNS:
            pronoun = w
            break

    if pronoun is None:
        return 0.0, 0.0

    correct_word = _extract_head_noun(correct)
    wrong_word   = _extract_head_noun(wrong)

    score_c = conceptnet_score(pronoun, correct_word, sentence)
    score_w = conceptnet_score(pronoun, wrong_word,   sentence)

    return score_c, score_w
