segment_seperator = "========"


def get_segment_seperator(level, name):
    return segment_seperator + "," + str(level) + "," + name

def get_seperator_foramt(levels=None):
    level_format = '\d' if levels is None else '[' + str(levels[0]) + '-' + str(levels[1]) + ']'
    separator_format = segment_seperator + ',' + level_format + ",.*?\\."
    return separator_format

def is_seperator_line(line):
    return line.startswith(segment_seperator)

def get_segment_level(separator_line):
    return int(separator_line.split(',')[1])

def get_segment_name(separator_line):
    return separator_line.split(',')[2]

def get_list_token():
    return "***LIST***"

def get_formula_token():
    return "***formula***"

def get_codesnipet_token():
    return "***codice***"

def get_special_tokens():
    special_tokens = [
        get_list_token(),
        get_formula_token(),
        get_codesnipet_token()
    ]
    return special_tokens