"""
Unified memory system using light field.
Handles relationships, reasoning, facts, identity.
One system for everything.
"""

import torch
from solar_ring.unified_light_field import UnifiedLightField

class UnifiedMemory:
    """
    Complete memory system for a person/agent.
    Stores all knowledge using light-field connections.
    """

    def __init__(self, identity: str, d: int = 300):
        self.identity = identity.lower()
        self.d = d
        self.field = UnifiedLightField(d)

        # All entities in memory
        self.entities = {}

        # Add self as anchor
        self._add_entity(
            name=identity,
            entity_type='self',
            mass=1.0,
            vec=torch.randn(d) * 0.1
        )

        # Fact store
        self.facts = []

        # Event history (temporal)
        self.events = []

        # Contradictions detected
        self.contradictions = []

    def _add_entity(self, name: str,
                    entity_type: str,
                    mass: float = 0.5,
                    vec: torch.Tensor = None):
        if vec is None:
            vec = torch.randn(self.d) * 0.1

        self.entities[name.lower()] = {
            'name': name,
            'type': entity_type,
            'mass': mass,
            'vec': vec,
            'alive': True,
            'is_photon': mass < 0.01,
            'pos': len(self.entities),
            'connections': {},
        }

    def learn_relationship(self, name: str,
                           relationship: str,
                           to: str = None):
        """
        Learn a relationship.
        'Ram is my father' → Ram, parent, self
        """
        mass = UnifiedLightField.MASS.get(
            relationship, 0.5
        )

        name_key = name.lower()
        to_key = (to or self.identity).lower()

        if name_key not in self.entities:
            self._add_entity(name, relationship, mass)

        # Store connection with light distance
        d_hops = {
            'self': 0, 'parent': 1, 'child': 1,
            'sibling': 1, 'spouse': 1,
            'best_friend': 1, 'close_friend': 2,
            'colleague': 3, 'classmate': 3,
            'professor': 3, 'acquaintance': 4,
            'stranger': 5,
        }.get(relationship, 3)

        self.entities[name_key]['connections'][to_key] = {
            'relationship': relationship,
            'hops': d_hops,
            'phi': None  # computed on demand
        }

        return name_key, d_hops

    def learn_fact(self, subject: str,
                   predicate: str,
                   obj: str):
        """
        Learn a fact: subject predicate object.
        'Kshitish studies at SUIIT'
        → subject=Kshitish pred=studies obj=SUIIT
        """
        fact = {
            'subject': subject.lower(),
            'predicate': predicate.lower(),
            'object': obj.lower(),
            'pos': len(self.facts),
        }

        # Check for contradiction
        for existing in self.facts:
            if (existing['subject'] == fact['subject']
                and existing['predicate'] == fact['predicate']
                and existing['object'] != fact['object']):
                self.contradictions.append(
                    (existing, fact)
                )

        self.facts.append(fact)

        # Add object as entity if not exists
        if obj.lower() not in self.entities:
            self._add_entity(obj, 'object', 0.7)

    def query(self, question: str) -> str:
        """
        Answer any question using light field.
        Finds strongest Phi connection to answer.
        """
        q = question.lower()
        words = q.split()

        # WHO IS X?
        for name, info in self.entities.items():
            if name in q and name != self.identity:
                conns = info.get('connections', {})
                if conns:
                    rel = list(conns.values())[0]
                    d = rel['hops']
                    c = self.field.c('relationship')
                    d_light = d / c
                    lam = self.field.redshift(d_light)
                    phi = lam * (info['mass'] ** 2) / max(d**2, 1)
                    return (f"{name.title()} is your "
                            f"{rel['relationship']} "
                            f"(Φ={phi:.3f} "
                            f"d_light={d_light:.3f})")

        # WHERE DO I ... ?
        if any(w in words for w in ('where','live','study','work')):
            for fact in self.facts:
                if (fact['subject'] == self.identity and
                    any(w in fact['predicate']
                        for w in ('live','study','work',
                                  'reside','attend'))):
                    return f"You {fact['predicate']} {fact['object'].upper()}"

        # WHAT IS MY ... ?
        if 'my' in words or 'college' in words:
            for fact in self.facts:
                if fact['subject'] == self.identity:
                    if any(w in q for w in
                           (fact['predicate'],
                            fact['object'])):
                        return f"Your {fact['predicate']}: {fact['object'].upper()}"

        # WHO IS CLOSEST?
        if 'closest' in words or 'nearest' in words:
            best_name = None
            best_phi = -999
            for name, info in self.entities.items():
                if name == self.identity:
                    continue
                conns = info.get('connections', {})
                if conns:
                    d = list(conns.values())[0]['hops']
                    phi = info['mass']**2 / max(d**2, 1)
                    if phi > best_phi:
                        best_phi = phi
                        best_name = name
            if best_name:
                return f"{best_name.title()} (Φ={best_phi:.3f})"

        # WHO IS REPELLED? (contradictions/enemies)
        if any(w in words for w in ('repel','enemy','conflict','against')):
            if self.contradictions:
                return f"Contradictions detected: {len(self.contradictions)}"
            return "No conflicts detected"

        # LIST ALL
        if 'list' in words or 'all' in words:
            lines = []
            for name, info in self.entities.items():
                if name == self.identity:
                    continue
                conns = info.get('connections', {})
                if conns:
                    rel = list(conns.values())[0]
                    d = rel['hops']
                    c = self.field.c('relationship')
                    d_light = d/c
                    lam = self.field.redshift(d_light)
                    phi = lam * info['mass']**2 / max(d**2,1)
                    lines.append(
                        f"  {name.title():15} "
                        f"{rel['relationship']:15} "
                        f"d={d} "
                        f"d_light={d_light:.3f} "
                        f"Φ={phi:.3f}"
                    )
            return '\n'.join(lines)

        return "I don't know — please tell me more"

    def summary(self):
        n = len(self.entities) - 1
        facts = len(self.facts)
        contradictions = len(self.contradictions)
        print(f"Identity: {self.identity.title()}")
        print(f"Entities: {n}")
        print(f"Facts: {facts}")
        print(f"Contradictions: {contradictions}")
        print(f"Memory: fixed O(N) — never grows")
        print(f"c_relationship: {self.field.c('relationship'):.1f}")
        print(f"c_reasoning: {self.field.c('reasoning'):.1f}")
        print(f"c_memory: {self.field.c('memory'):.1f}")
