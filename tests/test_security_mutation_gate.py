"""Characterize the bounded security mutation harness itself."""

from scripts.testing.run_security_mutation_gate import MUTATIONS


def test_security_mutations_have_unique_current_source_anchors() -> None:
    assert len(MUTATIONS) >= 8
    assert len({mutation.name for mutation in MUTATIONS}) == len(MUTATIONS)
    for mutation in MUTATIONS:
        content = __import__("pathlib").Path(mutation.path).read_text(encoding="utf-8")
        assert content.count(mutation.original) == 1, mutation.name
        assert mutation.original != mutation.replacement
        assert mutation.test.startswith("tests/")
