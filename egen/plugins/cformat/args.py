def render(arg, accept_def=True):
    if 'default' in arg and accept_def:
        defext = ' = {}'.format(arg['default'])
    else:
        defext = ''
    return '{dtype} {name}{defext}'.format(
        dtype = arg['type'],
        name = arg['name'],
        defext = defext)
