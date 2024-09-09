import pytest

from yava.tools import xmlfind


@pytest.mark.parametrize(
        ['xml', 'tag', 'content'], 
        [
            {'xml': "<PLAN>Content</PLAN>", 'tag': 'PLAN', 'content': 'Content'}
        ]
)
def test_single(xml, tag, content):

    content = xmlfind(xml, tag)
    assert content == content