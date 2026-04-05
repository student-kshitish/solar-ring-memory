"""
SubPlanetEnhanced: extended animacy, case, and physical-size detection.
Works alongside existing ring structure — does not replace it.
"""

ANIMACY_SIGNALS = {
    'human_male':   ['john', 'paul', 'tom', 'mike', 'bob',
                     'steve', 'chris', 'alex', 'sam', 'he',
                     'him', 'his', 'man', 'boy', 'father',
                     'son', 'brother', 'husband',
                     # expanded from Winograd schema names
                     'dave', 'david', 'george', 'jake', 'mark',
                     'nick', 'tim'],
    'human_female': ['mary', 'anna', 'lisa', 'sarah', 'beth',
                     'emma', 'diana', 'carol', 'she', 'her',
                     'hers', 'woman', 'girl', 'mother',
                     'daughter', 'sister', 'wife',
                     # expanded from Winograd schema names
                     'alice', 'amy', 'joan', 'rachel', 'sara',
                     'susan'],
    'animate':      ['cat', 'dog', 'bird', 'fish', 'animal',
                     'horse', 'cow', 'mouse', 'rabbit'],
    'inanimate':    ['trophy', 'suitcase', 'ball', 'window',
                     'car', 'tree', 'book', 'box', 'bottle',
                     'table', 'rock', 'cup', 'plate', 'chair',
                     'door', 'glass', 'phone', 'bag', 'coin'],
}

CASE_SIGNALS = {
    'nominative': ['he', 'she', 'they', 'who', 'i', 'we'],
    'accusative': ['him', 'her', 'them', 'whom', 'me', 'us'],
    'possessive': ['his', 'hers', 'their', 'whose', 'my', 'our'],
    'neuter':     ['it', 'its', 'which', 'that'],
}

SIZE_SIGNALS = {
    'large':   ['trophy', 'suitcase', 'car', 'tree', 'table',
                'door', 'bus', 'truck', 'building', 'house',
                'elephant', 'whale', 'mountain', 'ocean'],
    'small':   ['cup', 'coin', 'ring', 'button', 'seed',
                'ant', 'fly', 'pea', 'dot', 'chip', 'mouse'],
    'heavy':   ['trophy', 'rock', 'car', 'iron', 'anvil',
                'stone', 'brick', 'weight', 'safe', 'vault'],
    'light':   ['feather', 'paper', 'leaf', 'foam',
                'balloon', 'bubble', 'dust', 'air', 'cotton'],
    'fragile': ['window', 'glass', 'cup', 'vase', 'plate',
                'mirror', 'crystal', 'china', 'bulb'],
    'strong':  ['trophy', 'rock', 'car', 'iron', 'steel',
                'boulder', 'concrete', 'diamond'],
}

POS_MASS = {
    'SUBJ':  0.95,
    'OBJ':   0.90,
    'VERB':  0.85,
    'ADJ':   0.50,
    'ADV':   0.40,
    'PREP':  0.20,
    'CONJ':  0.15,
    'DET':   0.05,
    'OTHER': 0.10,
}


class SubPlanetEnhanced:
    """
    Enhanced sub-planet with parallel animacy, case, and size detection.
    Used alongside the existing ring structure — not replacing it.
    """

    def __init__(self):
        self.animacy = 'unknown'
        self.case    = 'unknown'
        self.size    = 'unknown'

    def update_parallel(self, token_text: str):
        """
        Detect animacy, case, and size simultaneously.
        Called for every token — adds structured knowledge
        on top of existing ring memory.
        """
        t = token_text.lower()

        # Sub-planet A: Animacy + Gender (independent)
        for category, words in ANIMACY_SIGNALS.items():
            if t in words:
                self.animacy = category
                break

        # Sub-planet B: Case (independent)
        for case, words in CASE_SIGNALS.items():
            if t in words:
                self.case = case
                break

        # Sub-planet C: Size / physical property (independent)
        for size, words in SIZE_SIGNALS.items():
            if t in words:
                self.size = size
                break

    def pronoun_compatibility(self, antecedent_text: str) -> float:
        """
        Score compatibility of this pronoun with an antecedent.
        Returns 0.0 to 1.0.
        Higher = more likely this pronoun refers to antecedent.

        Conservative scoring: only boost/penalize when the antecedent
        is definitively categorized. Unknown names stay at 0.5.
        """
        ant = antecedent_text.lower()
        score = 0.5  # neutral base

        human_male_set   = set(ANIMACY_SIGNALS.get('human_male',   []))
        human_female_set = set(ANIMACY_SIGNALS.get('human_female', []))
        animate_set      = set(ANIMACY_SIGNALS.get('animate',      []))
        inanimate_set    = set(ANIMACY_SIGNALS.get('inanimate',    []))
        all_animate      = human_male_set | human_female_set | animate_set

        # Neuter pronoun "it/its/which/that" → must be inanimate
        if self.case == 'neuter':
            if ant in inanimate_set:
                score += 0.4
            elif ant in all_animate:
                score -= 0.4
            # unknown → stay at 0.5

        # Nominative he/she/they → animate agreement
        elif self.case == 'nominative':
            if ant in all_animate:
                score += 0.2
            elif ant in inanimate_set:
                score -= 0.3
            # unknown → stay at 0.5

        # Gender agreement — only penalize KNOWN wrong-gender matches
        if self.animacy == 'human_male':
            if ant in human_male_set:
                score += 0.3   # confirmed match
            elif ant in human_female_set:
                score -= 0.5   # confirmed mismatch
            # ant unknown → no change

        if self.animacy == 'human_female':
            if ant in human_female_set:
                score += 0.3   # confirmed match
            elif ant in human_male_set:
                score -= 0.5   # confirmed mismatch
            # ant unknown → no change

        return max(0.0, min(1.0, score))

    def size_compatibility(self, context_adjective: str,
                           antecedent_text: str) -> float:
        """
        For Winograd schemas involving size/weight/fragility.
        e.g. "too big", "too heavy", "fragile"
        Returns boost if antecedent matches the property.
        """
        adj = context_adjective.lower()
        ant = antecedent_text.lower()

        SIZE_ADJ_MAP = {
            'big':    'large',
            'large':  'large',
            'huge':   'large',
            'small':  'small',
            'tiny':   'small',
            'little': 'small',
            'heavy':  'heavy',
            'light':  'light',
            'fragile':'fragile',
            'strong': 'strong',
            'weak':   None,
        }

        prop = SIZE_ADJ_MAP.get(adj)
        if prop is None:
            return 0.0
        if ant in SIZE_SIGNALS.get(prop, []):
            return 0.3
        return 0.0

    def describe(self) -> str:
        return (f"animacy={self.animacy} "
                f"case={self.case} "
                f"size={self.size}")


def build_sentence_sub_planets(sentence: str) -> dict:
    """
    Build a SubPlanetEnhanced for every word in sentence.
    Returns dict: word_index -> SubPlanetEnhanced
    """
    words = sentence.lower().split()
    sub_planets = {}
    for i, word in enumerate(words):
        sp = SubPlanetEnhanced()
        sp.update_parallel(word)
        sub_planets[i] = sp
    return sub_planets


def find_adjectives_in_context(sentence: str) -> list:
    """
    Find adjectives that might indicate size/weight/fragility.
    Returns list of adjective strings found.
    """
    SIZE_ADJS = {
        'big', 'large', 'huge', 'small', 'tiny', 'little',
        'heavy', 'light', 'fragile', 'strong', 'weak',
        'tall', 'short', 'long', 'thick', 'thin',
    }
    words = sentence.lower().split()
    return [w for w in words if w in SIZE_ADJS]
