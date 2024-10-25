from typing import Union

def object_to_xml(data: Union[dict, bool], root='object', ignore=None):
    if not ignore:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]
    xml = f'<{root}>'
    if isinstance(data, dict):
        for key, value in data.items():
            if not(key in ignore):
                xml += object_to_xml(value, key, ignore)

    elif isinstance(data, (list, tuple, set)):
        for item in data:
            xml += object_to_xml(item, 'item', ignore)

    else:
        xml += str(data)

    xml += f'</{root}>'
    return xml