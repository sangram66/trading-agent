"""
The pre-registration wall and the only surface agents may touch.

Two rules, enforced in code rather than in a prompt, because a prompt is a
suggestion and an autonomous loop will eventually route around a suggestion:

1. **No returns before a hypothesis.** `get_returns()` raises unless that
   hypothesis has already been registered with a mechanism, a predicted sign and
   magnitude, and a falsifier. You cannot look at the answer and then write down
   what you expected.

2. **A parameter tweak is a new hypothesis.** Re-testing H-0001 with a different
   z-threshold registers as H-0002 and costs another trial. This is what stops
   "iterate until it works" from being free, which is the mechanism by which
   automated research turns into automated self-deception.

Roles that must NOT be LLMs are not represented here at all — the risk officer,
the book, and the quant are deterministic modules the orchestrator calls. An LLM
that can be argued into approving a trade is not a risk officer.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = (
    "mechanism",            # why this happens, in prose
    "who_is_forced",        # which actor must trade regardless of price
    "counterparty",         # who takes the other side, and their compensation
    "predicted_sign",
    "predicted_magnitude_bp",
    "predicted_timing",
    "falsifier",            # what observation would kill this
)


class PreRegistrationError(RuntimeError):
    """Raised when an agent reaches for data it has not earned access to."""


@dataclass
class Hypothesis:
    hypothesis_id: str
    agent: str
    mechanism: str
    who_is_forced: str
    counterparty: str
    predicted_sign: str
    predicted_magnitude_bp: list
    predicted_timing: str
    falsifier: str
    parent_id: str | None = None
    registered_at: str = ""
    status: str = "registered"      # registered -> tested -> passed/failed
    verdict: dict | None = None


class HypothesisRegistry:
    def __init__(self, root: str | Path = "research/hypotheses"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, hid: str) -> Path:
        if not re.fullmatch(r"H-\d{4,}", hid):
            raise ValueError(f"bad hypothesis id {hid!r}; expected H-NNNN")
        return self.root / f"{hid}.json"

    def next_id(self) -> str:
        existing = [int(p.stem.split("-")[1]) for p in self.root.glob("H-*.json")]
        return f"H-{(max(existing) + 1 if existing else 1):04d}"

    def register(self, agent: str, parent_id: str | None = None, **fields) -> Hypothesis:
        """Commit a hypothesis to disk before any data is touched.

        Empty strings and placeholder text are rejected: an agent under pressure
        will happily write "TBD" in the falsifier field, and a falsifier of "TBD"
        is not a falsifier.
        """
        missing = [f for f in REQUIRED_FIELDS
                   if not fields.get(f) and fields.get(f) != 0]
        if missing:
            raise PreRegistrationError(
                f"cannot register: empty required field(s) {missing}. "
                f"A hypothesis without a mechanism and a falsifier is a guess.")
        for f in ("mechanism", "falsifier", "who_is_forced", "counterparty"):
            v = str(fields[f]).strip()
            if len(v) < 15 or v.lower() in {"tbd", "n/a", "unknown", "none"}:
                raise PreRegistrationError(
                    f"field {f!r} is a placeholder ({v!r}). State it properly.")

        hid = self.next_id()
        h = Hypothesis(hypothesis_id=hid, agent=agent, parent_id=parent_id,
                       registered_at=datetime.now(timezone.utc).isoformat(),
                       **{k: fields[k] for k in REQUIRED_FIELDS})
        self._path(hid).write_text(json.dumps(asdict(h), indent=2))
        return h

    def get(self, hid: str) -> Hypothesis:
        p = self._path(hid)
        if not p.exists():
            raise PreRegistrationError(f"{hid} is not registered")
        return Hypothesis(**json.loads(p.read_text()))

    def update(self, h: Hypothesis):
        self._path(h.hypothesis_id).write_text(json.dumps(asdict(h), indent=2))

    def all(self) -> list:
        return [Hypothesis(**json.loads(p.read_text()))
                for p in sorted(self.root.glob("H-*.json"))]


class AgentTools:
    """The complete surface an LLM agent can call. Nothing else is exposed.

    Constructed per-agent so every action carries an attributable name into the
    shared ledger. Agents share the ledger; they do not share a budget each.
    """

    def __init__(self, agent: str, ledger, registry: HypothesisRegistry,
                 returns_provider=None):
        self.agent = agent
        self.ledger = ledger
        self.registry = registry
        self._returns_provider = returns_provider
        self._unlocked: set = set()

    # -- step 1: you must write down what you expect --------------------
    def register_hypothesis(self, **fields) -> str:
        h = self.registry.register(self.agent, **fields)
        self._unlocked.add(h.hypothesis_id)
        return h.hypothesis_id

    def refine_hypothesis(self, parent_id: str, **fields) -> str:
        """A tweak is a new hypothesis and costs a new trial.

        Provided explicitly so that the honest path is also the easy path — if
        refinement were impossible, an agent would simply re-register the same
        idea and lose the lineage.
        """
        self.registry.get(parent_id)
        h = self.registry.register(self.agent, parent_id=parent_id, **fields)
        self._unlocked.add(h.hypothesis_id)
        return h.hypothesis_id

    # -- step 2: only now may you look ----------------------------------
    def get_returns(self, hypothesis_id: str, **kwargs) -> np.ndarray:
        if hypothesis_id not in self._unlocked:
            try:
                self.registry.get(hypothesis_id)
            except PreRegistrationError:
                raise PreRegistrationError(
                    f"{hypothesis_id} is not registered. Register a hypothesis "
                    f"— mechanism, who is forced, counterparty, predicted sign "
                    f"and magnitude, falsifier — before requesting any returns."
                ) from None
            raise PreRegistrationError(
                f"{hypothesis_id} was registered by another agent. Register "
                f"your own hypothesis rather than testing someone else's.")
        if self._returns_provider is None:
            raise RuntimeError("no returns provider configured")
        return self._returns_provider(hypothesis_id, **kwargs)

    # -- step 3: the gate charges you either way ------------------------
    def submit_for_verdict(self, hypothesis_id: str, strategy_returns):
        h = self.registry.get(hypothesis_id)
        v = self.ledger.record(hypothesis_id, self.agent, strategy_returns)
        h.status = "passed" if v.passed else "failed"
        h.verdict = asdict(v) if hasattr(v, "__dataclass_fields__") else dict(v)
        self.registry.update(h)
        return v

    # -- read-only situational awareness --------------------------------
    def budget_status(self) -> str:
        return self.ledger.summary()

    def journal(self) -> list:
        """Every hypothesis every agent has already killed.

        Exposed so agents stop re-proposing dead ideas. This is the only part of
        the design that actually makes the desk *self-improving* rather than
        merely parallel: the prior gets pruned, not just resampled.
        """
        return [{"id": h.hypothesis_id, "agent": h.agent, "status": h.status,
                 "mechanism": h.mechanism[:120],
                 "why_it_died": (h.verdict or {}).get("reason", "")}
                for h in self.registry.all() if h.status != "registered"]
