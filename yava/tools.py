import re

def xmlfind(text, tag):
    """Isolate the content of a tag from XML text.

    Arguments:
        text: The text to look in.
        tag: The tag to look for. In '<lat>23.7</lat>' the tag is 'lat'.

    Returns:
        A list containing the data in the requested tag.
    """
    rx = f'<{tag}>(.*?)</{tag}>'

    response = re.search(rx, text, re.DOTALL).group()
    response = response.strip(f'<{tag}>').strip(f'</{tag}>')
    
    return response