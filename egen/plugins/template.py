from string import Formatter

def build_template(template, funcs, *args):
    tmp_args = {}
    tmp_keys = [
        tmp_key[1]
        for tmp_key in Formatter().parse(template)
            if tmp_key[1] is not None
    ]
    for tmp_key in tmp_keys:
        handler_name = '_handle_' + tmp_key
        if handler_name not in funcs:
            raise Exception('Handler not found for template argument ' + tmp_key)
        tmp_args[tmp_key] = funcs[handler_name](*args)

    return template.format(**tmp_args)
