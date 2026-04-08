"""
Regression tests for escaping helpers and export_editors_notes rendering.

Covers the acceptance criteria from the issue:
  - Editor-notes exports render valid Markdown with special characters in
    report data (pipes, quotes, multi-line descriptions, markdown fences).
  - Mermaid relationship diagrams remain valid when names/labels contain
    punctuation or Mermaid-significant characters.
  - Tests cover escaping for tables, bullets, and Mermaid graph sections.
"""

import json
import re

import pytest

from novelforge.routes.export import (
    _md_text,
    _md_table_cell,
    _mermaid_node_label,
    _mermaid_edge_label,
)
from novelforge.progress import progress_manager


# ---------------------------------------------------------------------------
# Unit tests for escaping helpers
# ---------------------------------------------------------------------------

class TestMdText:
    """_md_text keeps content on a single logical Markdown line."""

    def test_passthrough_plain(self):
        assert _md_text("hello world") == "hello world"

    def test_newline_becomes_space(self):
        assert _md_text("line one\nline two") == "line one line two"

    def test_crlf_collapsed(self):
        assert _md_text("a\r\nb") == "a b"

    def test_cr_alone_becomes_space(self):
        # Bare CR (old Mac line-endings) is treated as a line break → space
        assert _md_text("a\rb") == "a b"

    def test_multiple_newlines(self):
        assert _md_text("a\n\nb\n\nc") == "a  b  c"

    def test_pipe_not_escaped(self):
        # _md_text does NOT escape pipes – callers that need table safety
        # should use _md_table_cell instead.
        assert "|" in _md_text("a | b")

    def test_non_string_coerced(self):
        assert _md_text(42) == "42"
        assert _md_text(None) == "None"


class TestMdTableCell:
    """_md_table_cell escapes pipe characters and removes newlines."""

    def test_passthrough_plain(self):
        assert _md_table_cell("normal text") == "normal text"

    def test_pipe_escaped(self):
        result = _md_table_cell("Alice | Bob")
        assert r"\|" in result
        assert result == r"Alice \| Bob"

    def test_multiple_pipes_escaped(self):
        result = _md_table_cell("a | b | c")
        assert result == r"a \| b \| c"

    def test_newline_becomes_space(self):
        assert _md_table_cell("line1\nline2") == "line1 line2"

    def test_crlf_collapsed(self):
        assert _md_table_cell("a\r\nb") == "a b"

    def test_pipe_and_newline_combined(self):
        result = _md_table_cell("ally | rival\nfriends")
        assert r"\|" in result
        assert "\n" not in result

    def test_non_string_coerced(self):
        assert _md_table_cell(0) == "0"


class TestMermaidNodeLabel:
    """_mermaid_node_label makes text safe inside Mermaid ["…"] syntax."""

    def test_passthrough_plain(self):
        assert _mermaid_node_label("Alice") == "Alice"

    def test_double_quote_replaced_with_entity(self):
        result = _mermaid_node_label('Al"ice')
        assert '"' not in result
        assert "#quot;" in result
        assert result == "Al#quot;ice"

    def test_newline_becomes_space(self):
        assert _mermaid_node_label("Alice\nBob") == "Alice Bob"

    def test_crlf_collapsed(self):
        assert _mermaid_node_label("Alice\r\nBob") == "Alice Bob"

    def test_cr_alone_becomes_space(self):
        assert _mermaid_node_label("Alice\rBob") == "Alice Bob"

    def test_pipe_left_intact(self):
        # Pipes are not special inside ["…"] node syntax
        assert "|" in _mermaid_node_label("Al|ice")

    def test_non_string_coerced(self):
        assert _mermaid_node_label(99) == "99"


class TestMermaidEdgeLabel:
    """_mermaid_edge_label makes text safe inside Mermaid -- "…" --> syntax."""

    def test_passthrough_plain(self):
        assert _mermaid_edge_label("loves") == "loves"

    def test_double_quote_replaced_with_entity(self):
        result = _mermaid_edge_label('enemy "forever"')
        assert '"' not in result
        assert "#quot;" in result

    def test_newline_becomes_space(self):
        assert _mermaid_edge_label("best\nfriend") == "best friend"

    def test_pipe_left_intact(self):
        # In the -- "…" --> form pipes are not delimiters; no escaping needed
        result = _mermaid_edge_label("a | b")
        assert result == "a | b"

    def test_non_string_coerced(self):
        assert _mermaid_edge_label(True) == "True"


# ---------------------------------------------------------------------------
# Helpers shared by integration tests
# ---------------------------------------------------------------------------

def _seed_token(token: str, **extra) -> None:
    """Seed the progress_manager with a completed generation entry."""
    progress_manager.create(token, {
        "status": "done",
        "current": 1,
        "total": 1,
        "step": "Complete",
        "error": None,
        "snapshot": {
            "title": "Test Novel",
            "genre": "Fantasy",
            "premise": "A quest.",
            "character_list": [{"name": "Alice", "role": "Hero"}],
        },
        "chapters_done": [
            {"number": 1, "title": "Ch1", "content": "Text.", "summary": "S1"},
        ],
        "consistency": {"overall_assessment": "Good.", "issues": []},
        **extra,
    })


def _export_notes(client, token: str, tmp_path, monkeypatch) -> str:
    """Call /export_editors_notes and return the file contents."""
    import novelforge.config as config
    monkeypatch.setattr(config, "EXPORT_DIR", str(tmp_path))

    r = client.post(
        "/export_editors_notes",
        data=json.dumps({"token": token}),
        content_type="application/json",
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    url = r.get_json()["download_url"]
    r2 = client.get(url)
    assert r2.status_code == 200
    return r2.data.decode("utf-8")


# ---------------------------------------------------------------------------
# Integration: Markdown table escaping (character relationship map)
# ---------------------------------------------------------------------------

class TestRelationshipTableEscaping:
    """Pipe characters in relationship data must not break Markdown table rows."""

    # A 4-column table row has 5 unescaped pipes: leading + 3 separators + trailing.
    _EXPECTED_ROW_PIPES = 5

    def test_pipe_in_character_name_escaped(self, client, tmp_path, monkeypatch):
        token = "00000000-0000-4000-8000-000000000050"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice | Wonderland", "Bob"],
            "relationships": [
                {"from": "Alice | Wonderland", "to": "Bob", "type": "friends", "label": ""},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # Every table row must have exactly 4 column-separator pipes
        table_rows = [
            line for line in content.splitlines()
            if line.startswith("| ") and not line.startswith("|---")
        ]
        # Skip the header row; check data rows
        for row in table_rows[1:]:
            assert row.count("|") - row.count(r"\|") == self._EXPECTED_ROW_PIPES, (
                f"Row has unexpected unescaped pipes: {row!r}"
            )

    def test_pipe_in_relationship_type_escaped(self, client, tmp_path, monkeypatch):
        token = "00000000-0000-4000-8000-000000000051"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice", "Bob"],
            "relationships": [
                {"from": "Alice", "to": "Bob", "type": "enemy | rival", "label": "tension"},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        assert r"\|" in content  # pipe was escaped somewhere in the table

    def test_pipe_in_description_escaped(self, client, tmp_path, monkeypatch):
        token = "00000000-0000-4000-8000-000000000052"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice", "Bob"],
            "relationships": [
                {
                    "from": "Alice",
                    "to": "Bob",
                    "type": "allied",
                    "label": "fought together | survived",
                },
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # Description cell must have escaped pipe
        assert r"\|" in content

    def test_newline_in_description_removed(self, client, tmp_path, monkeypatch):
        token = "00000000-0000-4000-8000-000000000053"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice", "Bob"],
            "relationships": [
                {
                    "from": "Alice",
                    "to": "Bob",
                    "type": "mentor",
                    "label": "taught her\nthe ways",
                },
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # Raw newline must not appear inside the table section
        table_section = content[content.find("| From |"):]
        assert "\n\n" not in table_section.split("```")[0]


# ---------------------------------------------------------------------------
# Integration: Mermaid diagram escaping
# ---------------------------------------------------------------------------

class TestMermaidEscaping:
    """Special characters in names/labels must not break Mermaid graph blocks."""

    def _get_mermaid_block(self, content: str) -> str:
        m = re.search(r"```mermaid\n(.*?)```", content, re.DOTALL)
        assert m, "No mermaid block found in output"
        return m.group(1)

    def test_double_quote_in_character_name_escaped(
        self, client, tmp_path, monkeypatch
    ):
        token = "00000000-0000-4000-8000-000000000054"
        _seed_token(token, character_relationship_map={
            "characters": ['Al"ice', "Bob"],
            "relationships": [
                {"from": 'Al"ice', "to": "Bob", "type": "rivals", "label": ""},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)
        block = self._get_mermaid_block(content)

        # Raw double-quote must not appear in node label position
        # (inside ["…"]) – only #quot; entity is allowed
        assert 'Al"ice' not in block
        assert "#quot;" in block

    def test_double_quote_in_edge_label_escaped(
        self, client, tmp_path, monkeypatch
    ):
        token = "00000000-0000-4000-8000-000000000055"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice", "Bob"],
            "relationships": [
                {"from": "Alice", "to": "Bob", "type": '"secret" allies', "label": ""},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)
        block = self._get_mermaid_block(content)

        # The edge label in -- "…" --> syntax must not contain raw "
        # (only the outer quotes from the syntax itself are allowed)
        edge_lines = [l for l in block.splitlines() if "--" in l and "-->" in l]
        assert edge_lines, "No edge lines found"
        for line in edge_lines:
            # Extract just the label portion between the outer quotes
            m = re.search(r'-- "(.*?)" -->', line)
            if m:
                assert '"' not in m.group(1), (
                    f"Unescaped quote in edge label: {line!r}"
                )

    def test_pipe_in_edge_label_does_not_use_pipe_syntax(
        self, client, tmp_path, monkeypatch
    ):
        """Edge labels with pipes must use -- "…" --> style, not -->|…| style."""
        token = "00000000-0000-4000-8000-000000000056"
        _seed_token(token, character_relationship_map={
            "characters": ["Alice", "Bob"],
            "relationships": [
                {"from": "Alice", "to": "Bob", "type": "friend | foe", "label": ""},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)
        block = self._get_mermaid_block(content)

        # The old -->|…| syntax must not be used for edges
        assert "-->|" not in block
        # The new -- "…" --> syntax must be used
        assert '-- "' in block

    def test_newline_in_character_name_removed(
        self, client, tmp_path, monkeypatch
    ):
        token = "00000000-0000-4000-8000-000000000057"
        _seed_token(token, character_relationship_map={
            "characters": ["Al\nice", "Bob"],
            "relationships": [
                {"from": "Al\nice", "to": "Bob", "type": "rivals", "label": ""},
            ],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)
        block = self._get_mermaid_block(content)

        # No raw newline inside the block (other than line separators)
        node_lines = [l for l in block.splitlines() if "[" in l and "]" in l]
        for line in node_lines:
            assert "\n" not in line  # each node is on one line already


# ---------------------------------------------------------------------------
# Integration: Markdown list-item escaping
# ---------------------------------------------------------------------------

class TestListItemEscaping:
    """Newlines in free-text list values must not break Markdown list structure."""

    def test_newline_in_consistency_issue_collapsed(
        self, client, tmp_path, monkeypatch
    ):
        token = "00000000-0000-4000-8000-000000000058"
        _seed_token(token, consistency={
            "overall_assessment": "Some issues.",
            "issues": ["Chapter 1 is\ngood\nbut chapter 2 is not"],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # The raw multi-line value must not appear verbatim
        assert "Chapter 1 is\ngood" not in content
        # A space-joined form should appear instead
        assert "Chapter 1 is good but chapter 2 is not" in content

    def test_pipe_in_list_item_passes_through(
        self, client, tmp_path, monkeypatch
    ):
        """Pipes in list items are left as-is (not a table context)."""
        token = "00000000-0000-4000-8000-000000000059"
        _seed_token(token, consistency={
            "overall_assessment": "Fine.",
            "issues": ["Chapters 1|2|3 need work"],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)
        assert "Chapters 1|2|3 need work" in content

    def test_markdown_fence_in_description_stays_on_one_line(
        self, client, tmp_path, monkeypatch
    ):
        """A value containing backtick fences doesn't open a new code block."""
        token = "00000000-0000-4000-8000-000000000060"
        hostile_thread = "```python\nprint('hello')\n```"
        _seed_token(token, loose_thread_report={
            "thread_integrity": "weak",
            "overall_assessment": "Some issues.",
            "unresolved_threads": [
                {
                    "thread": hostile_thread,
                    "chapters": [3],
                }
            ],
            "dangling_setup_elements": [],
            "intentionally_open_threads": [],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # The raw multi-line fence sequence from the hostile input must not be
        # preserved verbatim – newlines were collapsed to spaces.
        assert "```python\nprint" not in content
        # The collapsed (safe) form should appear on a single line within a list item.
        assert "```python print('hello') ```" in content

    def test_newline_in_recommendation_collapsed(
        self, client, tmp_path, monkeypatch
    ):
        token = "00000000-0000-4000-8000-000000000061"
        _seed_token(token, narrative_compression_report={
            "compression_priority": "medium",
            "overall_assessment": "OK",
            "redundant_sequences": [
                {
                    "chapters": [1, 2],
                    "pattern": "repetitive opening",
                    "recommendation": "Cut the first\nparagraph\nof each chapter.",
                }
            ],
            "emotional_beat_repetitions": [],
        })
        content = _export_notes(client, token, tmp_path, monkeypatch)

        # Raw newlines from recommendation must not appear
        assert "first\nparagraph" not in content
        # Collapsed form must be present
        assert "first paragraph of each chapter." in content
