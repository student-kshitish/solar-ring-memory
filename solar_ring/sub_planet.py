"""SubPlanet: three parallel sub-slots inside a noun ring."""

import torch


class SubPlanet:
    """
    Sub-planet inside a noun ring.
    Three sub-planets per noun — run in parallel.

    Sub-planet A: Quantity  (singular/plural/mass)
    Sub-planet B: Class     (animate/inanimate/abstract)
    Sub-planet C: Case      (subject/object/possessive)
    """

    QUANTITY_WORDS = {
        'singular': ['a', 'an', 'one', 'this', 'that'],
        'plural':   ['these', 'those', 'many', 'few', 'some', 'all'],
        'mass':     ['much', 'little', 'some', 'any'],
    }
    ANIMATE_WORDS = {
        'animate':   ['john', 'mary', 'tom', 'he', 'she', 'they',
                      'man', 'woman', 'boy', 'girl', 'cat', 'dog'],
        'inanimate': ['trophy', 'suitcase', 'ball', 'window', 'car',
                      'book', 'table', 'cup', 'plate', 'rock'],
        'abstract':  ['idea', 'concept', 'theory', 'belief', 'plan'],
    }

    def __init__(self, d_model: int, device: str = "cpu"):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.d = d_model
        self.device = device
        # A: quantity slot
        self.quantity = torch.zeros(d_model, device=device)
        self.quantity_label = 'unknown'
        # B: class slot
        self.animate_class = torch.zeros(d_model, device=device)
        self.animate_label = 'unknown'
        # C: case slot
        self.case = torch.zeros(d_model, device=device)
        self.case_label = 'unknown'

    def update_parallel(
        self,
        token_vec: torch.Tensor,
        token_text: str,
        pos_type: str,
    ):
        """
        Update all three sub-planets simultaneously.
        In Python this is sequential but conceptually parallel —
        each sub-planet operates on independent state with no
        data dependency between A, B, and C.
        """
        token_lower = token_text.lower()
        vec = token_vec.to(self.device)

        # Sub-planet A: Quantity (independent)
        for qty, words in self.QUANTITY_WORDS.items():
            if token_lower in words:
                self.quantity = vec.clone()
                self.quantity_label = qty
                break

        # Sub-planet B: Class (independent)
        for cls, words in self.ANIMATE_WORDS.items():
            if token_lower in words:
                self.animate_class = vec.clone()
                self.animate_label = cls
                break

        # Sub-planet C: Case (from POS, independent)
        if pos_type == 'SUBJ':
            self.case = vec.clone()
            self.case_label = 'nominative'
        elif pos_type == 'OBJ':
            self.case = vec.clone()
            self.case_label = 'accusative'

    def to_vector(self) -> torch.Tensor:
        """Concatenate all three sub-planet slots → (3·d_model,)."""
        return torch.cat([self.quantity, self.animate_class, self.case])

    def describe(self) -> str:
        return (f"qty={self.quantity_label} "
                f"class={self.animate_label} "
                f"case={self.case_label}")
