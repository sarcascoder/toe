"""Partial-order machinery for ambiguity-robust reading-order scoring.

Core idea
---------
The "true" reading order of a complex page is often NOT a single sequence.
Within a column, order is mandatory (paragraph 1 before paragraph 2). But two
independent columns, a body and a sidebar, or text and a footnote, may be read
in either relative order while still being "correct". Existing metrics compare
the model's linearization to ONE arbitrary gold sequence and unfairly penalize
valid alternative traversals.

We model the acceptable reading orders as a *partial order* P over blocks:
a set of required precedences (a must precede b). The set of all "correct"
linearizations is exactly the set of linear extensions of P. A model's
linearization is scored by how many *required* precedences it violates --
NOT by its distance to one chosen extension.

Definitions
-----------
constraints  : set of directed pairs (a, b) meaning "a must come before b".
closure(P)   : transitive closure -- all implied required precedences.
violation_rate(order, P):
    fraction of closure(P) pairs (a, b) that `order` contradicts
    (i.e. places b before a). 0.0 iff `order` is a valid linear extension.

Key provable properties (all unit-tested in tests/):
  P1 (ambiguity-robustness): violation_rate == 0 for EVERY linear extension
     of P, so independent regions incur no penalty.
  P2 (total-order reduction): if P is a total order, violation_rate equals the
     normalized Kendall-tau distance to that order.
  P3 (transcription-invariance): violation_rate depends only on block identities
     and their positions, never on block text content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations


@dataclass
class PartialOrder:
    n: int                                   # number of blocks (ids 0..n-1)
    constraints: set = field(default_factory=set)  # set[tuple[int,int]] a->b

    def add(self, a: int, b: int) -> "PartialOrder":
        """Require block a to precede block b."""
        if a == b:
            raise ValueError("a block cannot precede itself")
        self.constraints.add((a, b))
        return self

    def chain(self, ids) -> "PartialOrder":
        """Add sequential constraints across a list of block ids (i precedes i+1)."""
        ids = list(ids)
        for a, b in zip(ids, ids[1:]):
            self.add(a, b)
        return self

    def precede_all(self, group_a, group_b) -> "PartialOrder":
        """Every block in group_a must precede every block in group_b."""
        for a in group_a:
            for b in group_b:
                self.add(a, b)
        return self

    # ---- derived structures ----

    def closure(self) -> set:
        """Transitive closure of the constraint relation (Floyd-Warshall style)."""
        reach = {(a, b) for (a, b) in self.constraints}
        changed = True
        while changed:
            changed = False
            new = set()
            for (a, b) in reach:
                for (c, d) in reach:
                    if b == c and (a, d) not in reach:
                        new.add((a, d))
            if new:
                reach |= new
                changed = True
        return reach

    def is_consistent(self) -> bool:
        """No cycles (a must-precede-a would be contradictory)."""
        cl = self.closure()
        return not any(a == b for (a, b) in cl)

    def linear_extensions(self, limit: int | None = 5000):
        """Enumerate valid total orders (for testing / small n only)."""
        cl = self.closure()
        required = {(a, b) for (a, b) in cl}
        out = []
        for perm in permutations(range(self.n)):
            pos = {b: i for i, b in enumerate(perm)}
            if all(pos[a] < pos[b] for (a, b) in required):
                out.append(perm)
                if limit and len(out) >= limit:
                    break
        return out

    def num_required_pairs(self) -> int:
        return len(self.closure())


def violation_rate(order, P: PartialOrder) -> float:
    """Fraction of P's required precedences that `order` violates.

    `order` is a sequence of block ids (a linearization of a subset is allowed;
    pairs involving missing blocks are skipped, so this composes with detection
    metrics that handle coverage separately).
    """
    pos = {b: i for i, b in enumerate(order)}
    cl = P.closure()
    if not cl:
        return 0.0
    considered = 0
    violations = 0
    for (a, b) in cl:
        if a in pos and b in pos:
            considered += 1
            if pos[a] > pos[b]:   # b appears before a -> violation
                violations += 1
    if considered == 0:
        return 0.0
    return violations / considered


def order_consistency(order, P: PartialOrder) -> float:
    """1 - violation_rate. 1.0 = a valid reading; lower = more contradictions."""
    return 1.0 - violation_rate(order, P)


def kendall_tau_distance_normalized(order, reference) -> float:
    """Normalized Kendall-tau *distance* (0=identical, 1=reversed). Used to
    prove the total-order reduction property (P2)."""
    pos = {b: i for i, b in enumerate(order)}
    ref = list(reference)
    pairs = list(combinations(ref, 2))
    if not pairs:
        return 0.0
    disc = 0
    for a, b in pairs:                # a precedes b in reference
        if a in pos and b in pos and pos[a] > pos[b]:
            disc += 1
    return disc / len(pairs)
