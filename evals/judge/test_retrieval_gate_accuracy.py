"""LLM-AS-JUDGE EVAL — does the retrieval gate decide correctly?

The deterministic suite (evals/deterministic/test_retrieval_gate.py) pins the
plumbing: JSON parsing, fail-open posture, exactly-one-call. None of those
tests ask whether a "yes" was the *right answer*. This eval does.

We present the gate with messages paired with a memory context, each labeled
`should_retrieve: true | false`. A judge model scores whether the gate's
decision matches the expected label, covering the four cases that matter:

  1. Chitchat with a full memory store — should NOT retrieve
  2. A direct question about a stored fact — SHOULD retrieve
  3. A follow-up whose referent is only in history, not memory — should NOT
  4. A question whose answer memory does not contain — SHOULD retrieve

Requires the active provider's API key: the judge is a real model call, same
as evals/judge/test_response_quality.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from evals.helpers import HAS_KEY, response, text_block, ScriptedClient
from waku.memory.retrieval_gate import should_retrieve

pytestmark = pytest.mark.skipif(not HAS_KEY, reason="LLM-as-judge needs the active provider's API key")


# ──────────────────────────────────────────────────────────────────────
#  Dataset — each case is a message + the memory that exists at the time,
#  labeled with the correct gate decision. These are hand-curated to cover
#  the four categories from issue #77.
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GateCase:
    """A single labeled retrieval-gate test case.

    `message` is what the user says. `memory_snippets` is a short list of
    facts that exist in the store at the time (enough for the judge to
    reason about whether retrieval would help). `should_retrieve` is the
    ground-truth label.
    """
    id: str
    category: str
    message: str
    memory_snippets: list[str]
    should_retrieve: bool


DATASET: list[GateCase] = [
    # ── Category 1: Chitchat with a full memory store — should NOT retrieve
    GateCase(
        id="chitchat-1",
        category="chitchat",
        message="Hey, how's it going?",
        memory_snippets=[
            "User works at a startup called AlgoWars",
            "User prefers morning meetings",
            "User has a dog named Rex",
            "User is based in Bangalore",
        ],
        should_retrieve=False,
    ),
    GateCase(
        id="chitchat-2",
        category="chitchat",
        message="That's awesome lol",
        memory_snippets=[
            "User likes Rust and TypeScript",
            "User is building pods.ml",
            "User had coffee with Alex last Tuesday",
        ],
        should_retrieve=False,
    ),
    GateCase(
        id="chitchat-3",
        category="chitchat",
        message="Thanks! Appreciate it 🙏",
        memory_snippets=[
            "User's birthday is March 15",
            "User prefers concise responses",
            "User is 19 years old",
        ],
        should_retrieve=False,
    ),

    # ── Category 2: A direct question about a stored fact — SHOULD retrieve
    GateCase(
        id="direct-fact-1",
        category="direct-fact",
        message="When is my meeting with Alex?",
        memory_snippets=[
            "Meeting with Alex scheduled for Friday at 10am",
            "Alex prefers morning meetings",
        ],
        should_retrieve=True,
    ),
    GateCase(
        id="direct-fact-2",
        category="direct-fact",
        message="What's my dog's name?",
        memory_snippets=[
            "User has a dog named Rex",
            "User is based in Bangalore",
        ],
        should_retrieve=True,
    ),
    GateCase(
        id="direct-fact-3",
        category="direct-fact",
        message="Which language do I prefer for backend work?",
        memory_snippets=[
            "User prefers Rust for systems programming",
            "User uses TypeScript for frontend",
            "User is building AlgoWars",
        ],
        should_retrieve=True,
    ),

    # ── Category 3: A follow-up whose referent is only in history, not memory
    # The user is continuing a conversation, but the context is in the chat
    # history (which the gate does not see), not in the memory store. The
    # gate should NOT retrieve — the answer is in the conversation, not memory.
    GateCase(
        id="followup-history-1",
        category="followup-history",
        message="Can you make it earlier?",
        memory_snippets=[
            "User prefers morning meetings",
            "User has a dog named Rex",
        ],
        should_retrieve=False,
    ),
    GateCase(
        id="followup-history-2",
        category="followup-history",
        message="Yeah that one, tell me more about it",
        memory_snippets=[
            "User is building pods.ml",
            "User likes Rust",
        ],
        should_retrieve=False,
    ),
    GateCase(
        id="followup-history-3",
        category="followup-history",
        message="What about the second option?",
        memory_snippets=[
            "User is based in Bangalore",
            "User's birthday is March 15",
        ],
        should_retrieve=False,
    ),

    # ── Category 4: A question whose answer memory does not contain
    # The user asks something personal that Waku has no memory of. The gate
    # SHOULD retrieve — not because memory has the answer, but because the
    # gate's job is to decide "does this need memory?", not "will memory
    # succeed?". A personal question always warrants a search.
    GateCase(
        id="missing-memory-1",
        category="missing-memory",
        message="What did I have for breakfast yesterday?",
        memory_snippets=[
            "User is based in Bangalore",
            "User prefers morning meetings",
        ],
        should_retrieve=True,
    ),
    GateCase(
        id="missing-memory-2",
        category="missing-memory",
        message="Who was at the party last Saturday?",
        memory_snippets=[
            "User has a dog named Rex",
            "User works at AlgoWars",
        ],
        should_retrieve=True,
    ),
    GateCase(
        id="missing-memory-3",
        category="missing-memory",
        message="What's my friend Sarah's phone number?",
        memory_snippets=[
            "User prefers Rust",
            "User is 19 years old",
        ],
        should_retrieve=True,
    ),
]


# ──────────────────────────────────────────────────────────────────────
#  Scored eval — judge model decides if the gate's decision was correct
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def judge():
    """Build the same Anthropic-backed judge used by test_response_quality."""
    from evals.judge.anthropic_judge import AnthropicJudge
    return AnthropicJudge()


def _judge_gate_decision(judge, case: GateCase, gate_said: bool) -> tuple[bool, str]:
    """Ask the judge model: was the gate's decision correct for this case?

    Returns (correct, reason). The judge sees the user message, the memory
    that exists, the expected label, and the gate's actual decision. It
    returns a JSON verdict so we get a clean boolean + a short reason.
    """
    prompt = (
        "You are evaluating a retrieval gate for a personal assistant's memory.\n"
        "The gate decides whether to search the user's stored memories before\n"
        "generating a reply.\n\n"
        f"User message: {case.message!r}\n\n"
        f"Memories currently in the store:\n"
        + "\n".join(f"  - {m}" for m in case.memory_snippets)
        + f"\n\nExpected decision: {'retrieve' if case.should_retrieve else 'do NOT retrieve'}\n"
        f"Gate's actual decision: {'retrieve' if gate_said else 'do NOT retrieve'}\n\n"
        "Was the gate's decision correct? Consider:\n"
        "- If the message is chitchat/small-talk, the gate should NOT retrieve.\n"
        "- If the message directly asks about a stored fact, the gate SHOULD retrieve.\n"
        "- If the message is a follow-up whose context is in chat history (not\n"
        "  memory), the gate should NOT retrieve.\n"
        "- If the message asks something personal that memory doesn't contain,\n"
        "  the gate SHOULD still retrieve (the gate decides whether to search,\n"
        "  not whether the search will succeed).\n\n"
        'Reply with ONLY this JSON: {"correct": true/false, "reason": "<10 words>"}'
    )
    resp = judge.client.messages.create(
        model=judge.model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
    return verdict.get("correct", False), verdict.get("reason", "")


@pytest.mark.parametrize("case", DATASET, ids=[c.id for c in DATASET])
def test_gate_decision_is_correct(case: GateCase, judge):
    """For each labeled case, run the gate and ask the judge if the
    decision matches the expected label.

    This is a quality-scored eval, not a 0/1 assertion — the judge model
    decides whether the gate's choice was right, accounting for nuance that
    a simple label match would miss (e.g., a borderline chitchat that
    could reasonably go either way).
    """
    # Run the gate with the real model call
    from waku.config import load_settings
    from waku.loop.models import get_client

    settings = load_settings()
    client = get_client(settings)
    small_model = settings.small_model

    gate_said, _query, _reason = should_retrieve(client, small_model, case.message)

    # Judge whether the decision was correct
    correct, reason = _judge_gate_decision(judge, case, gate_said)

    assert correct, (
        f"Gate decided {'retrieve' if gate_said else 'do NOT retrieve'} "
        f"for case '{case.id}' ({case.category}), but expected "
        f"{'retrieve' if case.should_retrieve else 'do NOT retrieve'}. "
        f"Judge reason: {reason}"
    )
