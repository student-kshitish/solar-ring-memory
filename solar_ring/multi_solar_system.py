"""
MultiSolarSystem: manages multiple SolarMemory instances for long documents.
Each paragraph gets its own SolarMemory.
Systems communicate via gravitational waves (compressed Sun State vectors).

Does NOT modify existing SolarMemory class — works alongside it as a manager.
"""

import torch
from solar_ring.solar_memory import SolarMemory
from solar_ring.sun_state import SunState


class MultiSolarSystem:
    """
    Multiple solar systems for long documents.
    Each paragraph gets its own SolarMemory.
    Systems communicate via gravitational waves
    (compressed Sun State vectors).
    """

    def __init__(self, d_model: int, device,
                 alpha: float = 0.3,
                 max_systems: int = 10):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.d = d_model
        self.device = device
        self.alpha = alpha
        self.max_systems = max_systems

        self.systems: list       = []   # list of SolarMemory
        self.sun_states: list    = []   # list of SunState
        self.gravity_waves: list = []   # compressed vectors between systems
        self.active_idx = -1

        # Start first system
        self.new_paragraph()

    def new_paragraph(self):
        """
        Spawn new solar system for next paragraph.
        Pass gravitational wave from previous system.
        Returns the new SolarMemory.
        """
        if len(self.systems) >= self.max_systems:
            return self.systems[self.active_idx]

        new_mem = SolarMemory(device=self.device, dtype=torch.bfloat16)
        new_sun = SunState(self.d, alpha=self.alpha, device=self.device)

        # Inject gravity wave from previous system at half strength
        if self.gravity_waves:
            prev_wave = self.gravity_waves[-1]
            new_sun.state = prev_wave * 0.5

        self.systems.append(new_mem)
        self.sun_states.append(new_sun)
        self.active_idx = len(self.systems) - 1

        return new_mem

    @property
    def active(self) -> SolarMemory:
        return self.systems[self.active_idx]

    @property
    def active_sun(self) -> SunState:
        return self.sun_states[self.active_idx]

    def end_paragraph(self):
        """
        Called at end of each paragraph.
        Fuses ring summary vectors into active Sun State,
        then saves a compressed gravitational wave for the next system.
        """
        sun = self.active_sun
        mem = self.active

        planet_heads = []
        for ring in mem.rings:
            sv = ring.summary_vector()
            if sv.norm() > 0:
                planet_heads.append(sv)

        if planet_heads:
            sun.fuse(planet_heads)

        # Save compressed Sun as gravitational wave
        self.gravity_waves.append(sun.state.clone())

    def resolve_cross_paragraph(self,
                                pronoun_vec: torch.Tensor
                                ) -> torch.Tensor:
        """
        Resolve pronoun across ALL paragraph systems.
        Walk backward through systems with exponential decay.
        Returns resolved reference vector.
        """
        candidates = []
        weights = []

        for i, (sys, sun) in enumerate(zip(
            reversed(self.systems),
            reversed(self.sun_states)
        )):
            weight = 0.7 ** i  # exponential decay over recency

            # Local resolution within this system's ring hierarchy
            local = sys.resolve_pronoun(pronoun_vec)
            if local.norm() > 0:
                candidates.append(local)
                weights.append(weight)

            # Sun State resonance bonus
            res = sun.resonance(pronoun_vec)
            if res > 0.3:
                candidates.append(sun.state)
                weights.append(weight * res)

        if not candidates:
            return pronoun_vec

        total = sum(weights)
        if total < 1e-8:
            return candidates[0]

        resolved = sum(c * w / total for c, w in zip(candidates, weights))
        return resolved

    def get_resonance(self, token_vec: torch.Tensor) -> float:
        """
        Get resonance of token with ALL systems, weighted by recency.
        Recent systems have higher weight.
        """
        if not self.sun_states:
            return 0.0

        total_res = 0.0
        total_weight = 0.0
        for i, sun in enumerate(reversed(self.sun_states)):
            weight = 0.7 ** i
            res = sun.resonance(token_vec)
            total_res += res * weight
            total_weight += weight

        return total_res / max(total_weight, 1e-8)
