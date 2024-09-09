"""

    Collection of utilities

    xmlfind -> extract text in XML tags

"""


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

    response = re.findall(rx, text, re.DOTALL)

    if len(response) == 1:
        return response[0]
    else:
        return response
