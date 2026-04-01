"""POS/dependency tagger using spaCy, maps dep labels to role IDs."""

from .config import (
    ROLE_SUBJ, ROLE_OBJ, ROLE_VERB, ROLE_PREP,
    ROLE_CONJ, ROLE_ADJ, ROLE_DET, ROLE_OTHER,
    SPAWN_CONJUNCTIONS
)

# spaCy dependency label → role ID
DEP_TO_ROLE = {
    # Subject
    "nsubj": ROLE_SUBJ,
    "nsubjpass": ROLE_SUBJ,
    "csubj": ROLE_SUBJ,
    "csubjpass": ROLE_SUBJ,
    # Object
    "dobj": ROLE_OBJ,
    "obj": ROLE_OBJ,
    "iobj": ROLE_OBJ,
    "pobj": ROLE_OBJ,
    "attr": ROLE_OBJ,
    # Verb (ROOT and aux)
    "ROOT": ROLE_VERB,
    "root": ROLE_VERB,
    "aux": ROLE_VERB,
    "auxpass": ROLE_VERB,
    "xcomp": ROLE_CONJ,   # also spawns
    # Preposition
    "prep": ROLE_PREP,
    "agent": ROLE_PREP,
    # Conjunction / clause connectors (spawn triggers)
    "mark": ROLE_CONJ,
    "relcl": ROLE_CONJ,
    "advcl": ROLE_CONJ,
    "ccomp": ROLE_CONJ,
    "acl": ROLE_CONJ,
    "cc": ROLE_CONJ,
    # Adjective / modifier
    "amod": ROLE_ADJ,
    "advmod": ROLE_ADJ,
    "npadvmod": ROLE_ADJ,
    "acomp": ROLE_ADJ,
    # Determiner
    "det": ROLE_DET,
    "poss": ROLE_DET,
    "predet": ROLE_DET,
}


class POSTagger:
    """Wraps spaCy to produce per-token role IDs and spawn flags."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        import spacy
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import subprocess, sys
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True
            )
            self.nlp = spacy.load(model_name)

    def tag(self, text: str):
        """
        Returns list of dicts, one per token:
            {
                'text': str,
                'pos':  str,   # coarse POS
                'dep':  str,   # dependency label
                'role': int,   # mapped role ID
                'spawn': bool, # should this token trigger ring spawn?
            }
        """
        doc = self.nlp(text)
        result = []
        for token in doc:
            dep = token.dep_
            role = DEP_TO_ROLE.get(dep, ROLE_OTHER)
            spawn = dep in SPAWN_CONJUNCTIONS
            result.append({
                "text":  token.text,
                "pos":   token.pos_,
                "dep":   dep,
                "role":  role,
                "spawn": spawn,
                "idx":   token.i,
            })
        return result

    def tag_batch(self, texts: list[str]):
        return [self.tag(t) for t in texts]
