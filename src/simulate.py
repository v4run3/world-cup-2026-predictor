"""Step 7 — Tournament Monte-Carlo simulation.

Simulates the full World Cup 2026 thousands of times to estimate each team's
chance of reaching each round and winning the cup.

Method
------
* **Groups** are inferred from the real fixture list (12 groups of 4).
* **Scorelines** are sampled from the Poisson goal model (independent Poisson
  draws for each side), which naturally yields points, goal difference, and
  knockout results.
* **Qualification:** top 2 of each group + the 8 best third-placed teams = 32.
* **Knockout:** the 32 qualifiers are seeded (group winners > runners-up >
  thirds, then by points/GD/GF) into a standard single-elimination bracket, so
  the two top seeds can only meet in the final. Draws in knockouts go to a
  penalty shootout decided by relative expected goals.

The seeded bracket is a transparent approximation of FIFA's official (and far
more intricate) slot-mapping — it preserves the key property that stronger teams
have easier early paths, which is what matters for champion odds.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

import src.data_collection as dc
from src.predict import match_lambdas

ROUND_NAMES = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"]


def infer_groups(fixtures: pd.DataFrame) -> dict[str, list[str]]:
    """Recover the 12 groups as connected components of the fixture graph."""
    adj: dict[str, set] = defaultdict(set)
    for r in fixtures.itertuples(index=False):
        adj[r.home_team].add(r.away_team)
        adj[r.away_team].add(r.home_team)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        comps.append(sorted(comp))
    comps.sort(key=lambda g: g[0])
    return {chr(65 + i): g for i, g in enumerate(comps)}


def _bracket_seed_order(n: int) -> list[int]:
    """Standard single-elim seed positions (1-indexed) for a bracket of size n."""
    if n == 1:
        return [1]
    prev = _bracket_seed_order(n // 2)
    order = []
    for s in prev:
        order.append(s)
        order.append(n + 1 - s)
    return order


class TournamentSimulator:
    """Pre-computes goal expectations once, then runs many fast simulations."""

    def __init__(self, fixtures: pd.DataFrame | None = None):
        if fixtures is None:
            fixtures = dc.load_fixtures_2026()
        self.groups = infer_groups(fixtures)
        self.teams = [t for g in self.groups.values() for t in g]
        # Pre-compute neutral-venue lambdas for every ordered pair (fast lookups).
        self._lam: dict[tuple[str, str], tuple[float, float]] = {}
        for a in self.teams:
            for b in self.teams:
                if a != b:
                    self._lam[(a, b)] = match_lambdas(a, b, neutral=True)

    # --- single match ------------------------------------------------------
    def _play(self, rng, a: str, b: str, knockout: bool):
        """Return (winner, goals_a, goals_b). Knockout ties go to penalties."""
        lam_a, lam_b = self._lam[(a, b)]
        ga, gb = rng.poisson(lam_a), rng.poisson(lam_b)
        if ga > gb:
            return a, ga, gb
        if gb > ga:
            return b, ga, gb
        if not knockout:
            return None, ga, gb  # draw allowed
        # penalty shootout: bias by relative attacking strength
        p_a = lam_a / (lam_a + lam_b)
        return (a if rng.random() < p_a else b), ga, gb

    # --- one full tournament ----------------------------------------------
    def _simulate_once(self, rng) -> tuple[str, dict[str, int]]:
        """Return (champion, {team: deepest round index reached})."""
        reached: dict[str, int] = {}
        standings = {}  # team -> [pts, gd, gf]
        winners, runners, thirds = [], [], []

        for label, teams in self.groups.items():
            table = {t: [0, 0, 0] for t in teams}
            for a, b in combinations(teams, 2):
                _, ga, gb = self._play(rng, a, b, knockout=False)
                table[a][0] += 3 if ga > gb else (1 if ga == gb else 0)
                table[b][0] += 3 if gb > ga else (1 if ga == gb else 0)
                table[a][1] += ga - gb
                table[b][1] += gb - ga
                table[a][2] += ga
                table[b][2] += gb
            # Rank the group ONCE; reuse for winner/runner/third (avoids dupes).
            ranked = sorted(teams, key=lambda t: (table[t][0], table[t][1], table[t][2], rng.random()), reverse=True)
            standings.update({t: table[t] for t in teams})
            winners.append(ranked[0])
            runners.append(ranked[1])
            thirds.append(ranked[2])
            reached[ranked[0]] = reached[ranked[1]] = 0  # top 2 qualify

        # 8 best third-placed teams qualify too.
        tier_sort = lambda lst: sorted(lst, key=lambda t: (standings[t][0], standings[t][1], standings[t][2]), reverse=True)
        qualified_thirds = tier_sort(thirds)[:8]
        for t in qualified_thirds:
            reached[t] = 0

        # Seed the 32: winners, then runners-up, then qualified thirds.
        seeded = tier_sort(winners) + tier_sort(runners) + qualified_thirds  # 32 teams

        # Arrange into a standard 32-slot bracket.
        order = _bracket_seed_order(32)
        bracket = [seeded[s - 1] for s in order]

        # Single elimination.
        round_idx = 1
        current = bracket
        while len(current) > 1:
            nxt = []
            for i in range(0, len(current), 2):
                w, _, _ = self._play(rng, current[i], current[i + 1], knockout=True)
                nxt.append(w)
                reached[w] = round_idx
            current = nxt
            round_idx += 1
        return current[0], reached

    # --- many tournaments --------------------------------------------------
    def run(self, n_sims: int = 2000, seed: int = 12345) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = {t: np.zeros(len(ROUND_NAMES)) for t in self.teams}
        for _ in range(n_sims):
            _champion, reached = self._simulate_once(rng)
            for t, deepest in reached.items():
                # team reached every round up to `deepest` (deepest 5 == champion)
                for r in range(deepest + 1):
                    counts[t][r] += 1

        rows = []
        for t in self.teams:
            c = counts[t] / n_sims
            rows.append({
                "team": t,
                "group": self._group_of(t),
                "Reach R16 %": round(c[1] * 100, 1),
                "Reach QF %": round(c[2] * 100, 1),
                "Reach SF %": round(c[3] * 100, 1),
                "Reach Final %": round(c[4] * 100, 1),
                "Win Cup %": round(c[5] * 100, 1),
            })
        return pd.DataFrame(rows).sort_values("Win Cup %", ascending=False).reset_index(drop=True)

    def _group_of(self, team: str) -> str:
        for label, teams in self.groups.items():
            if team in teams:
                return label
        return "?"


if __name__ == "__main__":
    sim = TournamentSimulator()
    print(f"Simulating with {len(sim.teams)} teams in {len(sim.groups)} groups...")
    table = sim.run(n_sims=2000)
    print("\nTop 15 title contenders:")
    print(table.head(15).to_string(index=False))
