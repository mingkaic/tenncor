from plugins.template import build_template
from plugins.common import strip_template_prefix
from plugins.cformat.args import render as arender

_decl_template = '''
{comment}
{template_decl}{outtype} {funcname} ({args});
'''

_defn_template = '''
{template_decl}{outtype} {funcname} ({args})
{{
    if ({null_check})
    {{
        global::fatal("cannot {funcname} with a null argument");
    }}
    {block}
}}
'''

def _handle_comment(obj):
    comment = obj.get('description', _handle_funcname(obj) + ' ...')
    comment_lines = comment.split('\n')
    return '\n'.join(['/// ' + line for line in comment_lines])

def _handle_template_decl(obj):
    temp = obj.get('template', '').strip()
    if len(temp) > 0:
        return 'template <{}>\n'.format(temp)
    return ''

def _handle_outtype(obj):
    assert('out' in obj)
    otype = None
    out = obj['out']
    if isinstance(out, dict):
        otype = out.get('type', None)
    if 'operator' in obj and otype is None:
        # account for type conversion operator
        return ''
    if otype is None:
        return 'void'
    return otype

def _handle_funcname(obj):
    assert(('name' in obj) != ('operator' in obj))
    if 'name' in obj:
        return obj['name']
    elif 'operator' in obj:
        return 'operator ' + obj['operator']
    else:
        raise Exception('No name or operator specified in func')

def _handle_null_check(obj):
    args = obj.get('args', [])
    tens = list(filter(lambda arg: arg.get('check_null', True) and (
        'teq::TensptrT' in arg['type'] or
        'eteq::ETensor<T>' in arg['type']), args))
    if len(tens) == 0:
        return 'false'
    varnames = [ten['name'] for ten in tens]
    return ' || '.join([varname + ' == nullptr' for varname in varnames])

def _handle_block(obj):
    assert('out' in obj)
    outval = obj['out']
    if isinstance(outval, dict):
        assert('val' in outval)
        outval = outval['val']
    return outval

def render_args(obj, is_decl):
    return ', '.join([
        arender(arg, accept_def=is_decl)
        for arg in obj.get('args', [])
    ])

def render_decl(obj):
    handlers = dict(globals())
    handlers['_handle_args'] = lambda obj: render_args(obj, True)
    return build_template(_decl_template, handlers, obj)

def render_defn(obj, clas=None):
    handlers = dict(globals())
    if clas is not None:
        temps = clas.get('template', '')
        if len(temps) > 0:
            handlers['_handle_template_decl'] = lambda obj: 'template <{}>\n'.format(temps)
            clean_temp = ','.join([strip_template_prefix(t) for t in temps.split(',')])
            func_prefix = '{}<{}>::'.format(clas['name'], clean_temp)
        else:
            func_prefix = clas['name'] + '::'
        handlers['_handle_funcname'] = lambda obj: func_prefix + _handle_funcname(obj)
    handlers['_handle_args'] = lambda obj: render_args(obj, False)
    return build_template(_defn_template, handlers, obj)
