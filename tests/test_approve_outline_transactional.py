"""
Tests that verify approve_outline() is transactional:
a planning-agent failure during the approval must leave the session
in its original state (no partial mutations).
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAPTERS = [{"number": 1, "title": "The Start", "summary": "It begins."}]
_CHARACTERS = [
    {
        "name": "Alice",
        "age": "30",
        "role": "Hero",
        "background": "A wanderer.",
        "arc": "Learns to lead.",
    }
]

_SESSION_SEED = {
    "title": "Original Title",
    "premise": "An original premise.",
    "genre": "Fantasy",
    "chapters": 1,
    "word_count": 50000,
    "special_events": "",
    "special_instructions": "Keep it short.",
    "chapter_list": [{"number": 1, "title": "Old Chapter", "summary": "Old summary."}],
    "character_list": [
        {
            "name": "Alice",
            "age": "28",
            "role": "Protagonist",
            "background": "Background.",
            "arc": "Arc.",
        }
    ],
    "narrative_perspective": "third_person",
    "story_architecture": {"old": "arch"},
    "master_timeline": {"old": "timeline"},
    "character_fate_registry": {"old": "fate"},
    "character_arc_plan": {"old": "arc"},
    "antagonist_motivation_plan": {"old": "antag"},
    "technology_rules": {"old": "tech"},
    "theme_reinforcement": {"old": "theme"},
    "pov_focal_character_plan": {"old": "pov"},
    "_agent_input_hashes": {},
}


def _seed_session(client, overrides: dict | None = None) -> None:
    """Populate the test client session with baseline data."""
    with client.session_transaction() as sess:
        for key, value in _SESSION_SEED.items():
            sess[key] = value
        if overrides:
            for key, value in overrides.items():
                sess[key] = value


def _read_session(client) -> dict:
    """Return a snapshot of every key stored in the current session."""
    snapshot: dict = {}
    with client.session_transaction() as sess:
        for key in list(sess.keys()):
            snapshot[key] = sess[key]
    return snapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_client(client, mock_llm):
    """A test client whose session is pre-seeded with valid outline data."""
    _seed_session(client)
    return client


# ---------------------------------------------------------------------------
# Helpers that construct the request payload
# ---------------------------------------------------------------------------

def _approve_payload(
    title: str = "New Title",
    chapters: list | None = None,
    characters: list | None = None,
    narrative_perspective: str = "third_person",
) -> dict:
    return {
        "title": title,
        "chapters": chapters if chapters is not None else _CHAPTERS,
        "characters": characters if characters is not None else _CHARACTERS,
        "narrative_perspective": narrative_perspective,
    }


# ---------------------------------------------------------------------------
# Happy-path sanity check
# ---------------------------------------------------------------------------

class TestApproveOutlineTransactionalHappyPath:
    def test_success_commits_all_fields(self, seeded_client):
        """A successful approval must update title, chapters, characters, and
        planning-agent outputs in the session."""
        r = seeded_client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload(title="New Title")),
            content_type="application/json",
        )
        assert r.status_code == 200
        assert r.get_json()["status"] == "approved"

        snap = _read_session(seeded_client)
        assert snap["title"] == "New Title"
        assert snap["chapter_list"] == [
            {"number": 1, "title": "The Start", "summary": "It begins."}
        ]
        assert snap["character_list"][0]["name"] == "Alice"
        # Planning agent hashes must be persisted
        assert "_agent_input_hashes" in snap


# ---------------------------------------------------------------------------
# Transactionality: Group 1 failure
# ---------------------------------------------------------------------------

class TestApproveOutlineTransactionalGroup1Failure:
    def test_group1_failure_leaves_session_intact(self, client, mocker, mock_llm):
        """When a Group-1 planning agent raises, the session must not be mutated."""
        _seed_session(client)
        before = _read_session(client)

        # Make plan_story_architecture raise after being called
        mocker.patch(
            "novelforge.routes.outline.plan_story_architecture",
            side_effect=RuntimeError("LLM unavailable"),
        )

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload(title="Should Not Stick")),
            content_type="application/json",
        )
        # The route must return an error (500 from the unhandled RuntimeError)
        assert r.status_code == 502

        after = _read_session(client)
        # Session must be identical to pre-request state
        assert after["title"] == before["title"], (
            "title was mutated before the agent failure"
        )
        assert after["chapter_list"] == before["chapter_list"], (
            "chapter_list was mutated before the agent failure"
        )
        assert after["character_list"] == before["character_list"], (
            "character_list was mutated before the agent failure"
        )
        assert after["story_architecture"] == before["story_architecture"], (
            "story_architecture was partially updated"
        )
        assert after["_agent_input_hashes"] == before["_agent_input_hashes"], (
            "_agent_input_hashes was updated despite failure"
        )

    def test_group1_master_timeline_failure_leaves_session_intact(self, client, mocker, mock_llm):
        """plan_master_timeline failure must not leave partial session state."""
        _seed_session(client)
        before = _read_session(client)

        mocker.patch(
            "novelforge.routes.outline.plan_master_timeline",
            side_effect=RuntimeError("Timeout"),
        )

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload()),
            content_type="application/json",
        )
        assert r.status_code == 502

        after = _read_session(client)
        assert after["title"] == before["title"]
        assert after["master_timeline"] == before["master_timeline"]
        assert after["narrative_perspective"] == before["narrative_perspective"]


# ---------------------------------------------------------------------------
# Transactionality: Group 2 failure
# ---------------------------------------------------------------------------

class TestApproveOutlineTransactionalGroup2Failure:
    def test_group2_failure_leaves_session_intact(self, client, mocker, mock_llm):
        """When a Group-2 planning agent raises, no session field must change."""
        _seed_session(client)
        before = _read_session(client)

        mocker.patch(
            "novelforge.routes.outline.plan_character_arc_plan",
            side_effect=RuntimeError("Context window exceeded"),
        )

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload()),
            content_type="application/json",
        )
        assert r.status_code == 502

        after = _read_session(client)
        assert after["title"] == before["title"]
        assert after["chapter_list"] == before["chapter_list"]
        assert after["character_arc_plan"] == before["character_arc_plan"]
        # Group 1 agents ran and produced NEW results — but must NOT be in session
        assert after["story_architecture"] == before["story_architecture"]
        assert after["master_timeline"] == before["master_timeline"]

    def test_group2_antagonist_failure_leaves_session_intact(self, client, mocker, mock_llm):
        """plan_antagonist_motivation_plan failure must not leave partial state."""
        _seed_session(client)
        before = _read_session(client)

        mocker.patch(
            "novelforge.routes.outline.plan_antagonist_motivation_plan",
            side_effect=RuntimeError("Rate limit"),
        )

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload()),
            content_type="application/json",
        )
        assert r.status_code == 502

        after = _read_session(client)
        assert after["antagonist_motivation_plan"] == before["antagonist_motivation_plan"]
        assert after["character_list"] == before["character_list"]


# ---------------------------------------------------------------------------
# Transactionality: Group 3 failure
# ---------------------------------------------------------------------------

class TestApproveOutlineTransactionalGroup3Failure:
    def test_group3_failure_leaves_session_intact(self, client, mocker, mock_llm):
        """When plan_pov_focal_character raises, no session field must change."""
        _seed_session(client)
        before = _read_session(client)

        mocker.patch(
            "novelforge.routes.outline.plan_pov_focal_character",
            side_effect=RuntimeError("Network error"),
        )

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload()),
            content_type="application/json",
        )
        assert r.status_code == 502

        after = _read_session(client)
        assert after["pov_focal_character_plan"] == before["pov_focal_character_plan"]
        # Even title / chapters must remain unchanged
        assert after["title"] == before["title"]
        assert after["chapter_list"] == before["chapter_list"]
        # Group 1 + 2 ran successfully but must NOT be persisted
        assert after["story_architecture"] == before["story_architecture"]
        assert after["character_arc_plan"] == before["character_arc_plan"]
        assert after["_agent_input_hashes"] == before["_agent_input_hashes"]


# ---------------------------------------------------------------------------
# Transactionality: rename propagation is not applied on failure
# ---------------------------------------------------------------------------

class TestApproveOutlineTransactionalRenamePropagation:
    def test_rename_not_persisted_on_agent_failure(self, client, mocker, mock_llm):
        """Character renames must only be committed if all agents succeed."""
        _seed_session(client, overrides={"premise": "Alice sets out on a journey."})
        before = _read_session(client)

        mocker.patch(
            "novelforge.routes.outline.plan_story_architecture",
            side_effect=RuntimeError("Failure"),
        )

        renamed_chars = [
            {
                "name": "Bob",  # renamed from Alice
                "age": "30",
                "role": "Hero",
                "background": "A wanderer.",
                "arc": "Learns to lead.",
            }
        ]

        r = client.post(
            "/approve_outline",
            data=json.dumps(_approve_payload(characters=renamed_chars)),
            content_type="application/json",
        )
        assert r.status_code == 502

        after = _read_session(client)
        # premise must still contain the original name
        assert after["premise"] == before["premise"]
        assert after["character_list"] == before["character_list"]
