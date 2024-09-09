import pytest

from yava.tools import xmlfind


@pytest.mark.parametrize(
        ['xml', 'tag', 'content'], 
        [
            # Standard, easy
            ("<PLAN>Content</PLAN>", 'PLAN', 'Content'),
            # Tag does not exists
            ("<PLA>Content</PLAN>", "PLAN", []),
            # Test multiple instances
            ("<PLAN>Plan 1</PLAN><PLAN>Plan 2</PLAN>", "PLAN", ["Plan 1", "Plan 2"]),
        ]
)
def test_single(xml, tag, content):

    response = xmlfind(xml, tag)
    assert content == response