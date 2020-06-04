import re

from plugins.common import strip_template_prefix

_template = '''{mod}.def("{fname}", {func});'''

_py_op = {
    ('-', 1): '__neg__',
    ('+', 2): '__add__',
    ('*', 2): '__mul__',
    ('-', 2): '__sub__',
    ('/', 2): '__truediv__',
    ('==', 2): '__eq__',
    ('!=', 2): '__ne__',
    ('<', 2): '__lt__',
    ('>', 2): '__gt__',
}

_py_op_rev = {
    '+': '__radd__',
    '*': '__rmul__',
    '-': '__rsub__',
    '/': '__rtruediv__',
}

pybindt = 'PybindT'

def _sub_pybind(stmt, source):
    type_pattern = '([^\\w]){}([^\\w])'.format(source)
    type_replace = '\\1{}\\2'.format(pybindt)
    return re.sub(type_pattern, type_replace, ' ' + stmt + ' ').strip()

def _render_pyarg(arg, templates):
    affix = ''
    if arg.get('default', None) is not None:
        affix = '=' + clean_templates(str(arg['default']), templates)
    return 'py::arg("{name}"){affix}'.format(name=arg['name'], affix=affix)

def _clean_type(atype):
    segs = re.split(r'\*|\&|\s', atype)
    return ' '.join(filter(lambda s: len(s) > 0 and s != 'const', segs))

def _defn_type(atype, mod):
    mname = process_modname(atype)
    return 'py::class_<{fulltype}> {modname}({mod}, "{strname}");'.format(
        fulltype=atype,
        modname=mname,
        mod=mod,
        strname=mname[2:])

def _defn_intypes(args, mod):
    func_inputs = [_clean_type(a['type']) for a in args]
    return dict([(process_modname(atype), _defn_type(atype, mod)) for atype in func_inputs])

def _render_func(params, args, block, description):
    description = '\n'.join(['"' + line + '"' for line in description.split('\n')])
    func = '[]({params}) {{{block}}}'.format(params=params, block=block)
    return ', '.join(filter(lambda s: len(s) > 0, [func, args, description]))

def clean_templates(s, templates):
    for typenames in templates:
        s = _sub_pybind(s, strip_template_prefix(typenames))
    return s

def process_modname(atype):
    # trim off template and namespace
    endtype = atype.split('<', 1)[0].split('::')[-1]
    return 'm_' + endtype

def render_classfunc(obj, class_type, namespace, mod):
    templates = class_type.get('template', '').split(',')
    cname = 'cls_' + class_type['name']

    assert('out' in obj)
    otype = None
    out = obj['out']
    if isinstance(out, dict):
        otype = out.get('type', None)
    if otype is None and 'operator' in obj:
        raise Exception('conversion operator not supported for inline functions')

    # process args
    args = obj.get('args', [])
    # common template cleanup for all args
    for arg in args:
        arg['type'] = clean_templates(arg['type'], templates)

    # auto-define input types
    func_inputs = _defn_intypes(args, mod)

    selftype = namespace + "::" + class_type['name']
    if 'template' in class_type:
        selftype += '<{}>'.format(pybindt)
    fullargs = [{
            'name': 'self',
            'type': selftype + '&',
            'default': None,
        }] + args

    name = obj.get('name', obj.get('operator', ''))
    comment = obj.get('description', name + ' ...')

    params = ', '.join([arg['type'] + ' ' + arg['name'] for arg in fullargs])
    pargs = ', '.join([_render_pyarg(arg, templates) for arg in args])

    # process name
    assert(('name' in obj) != ('operator' in obj))
    if 'name' in obj:
        block = 'return self.{name}({args});'.format(name=name,
            args=', '.join([arg['name'] for arg in args]))
        return _template.format(mod=cname, fname=name,
            func=_render_func(params, pargs, block, comment)), func_inputs

    if 'operator' not in obj:
        raise Exception('No name or operator specified in func')

    funcs = []
    op = obj['operator']
    if len(args) > 0 and otype not in args[0]['type'] and op in _py_op_rev:
        name = _py_op_rev[op]
        if len(fullargs) == 1:
            block = 'return ' + op + fullargs[0]
        else:
            block = 'return ' + op.join(fullargs[::-1])
        block = ';'
        funcs.append(_template.format(mod=cname, fname=name,
            func=_render_func(params, pargs, block, comment)))

    if len(fullargs) == 1:
        block = 'return ' + op + fullargs[0]
    else:
        block = 'return ' + op.join(fullargs)
    block = ';'
    name = _py_op[(op, 1 + len(args))]
    funcs.append(_template.format(mod=cname, fname=name,
        func=_render_func(params, pargs, block, comment)))

    return '\n'.join(funcs), func_inputs

def render(obj, mod, namespace):
    templates = obj.get('template', '').split(',')

    assert('out' in obj)
    otype = None
    out = obj['out']
    if isinstance(out, dict):
        otype = out.get('type', None)
    if otype is None:
        raise Exception('conversion operator not supported for inline functions')
    otype = clean_templates(otype, templates)

    # process args
    params = obj.get('args', [])
    # common template cleanup for all args
    for param in params:
        param['type'] = clean_templates(param['type'], templates)

    # auto-define input types
    func_inputs = _defn_intypes(params, mod)

    # process name
    assert(('name' in obj) != ('operator' in obj))
    if 'name' in obj:
        name = obj['name']
        block = 'return {name}({args});'.format(name=namespace + "::" + name,
            args=', '.join([param['name'] for param in params]))
        args = params
    elif 'operator' in obj:
        op = obj['operator']
        # operators are made members of the output type,
        # since conversions aren't supported (otherwise use first non-primitive input type)
        mod = process_modname(otype)
        # inline conversions not supported, all non-conversions have output type as one of the inputs
        assert(mod in func_inputs)

        if len(params) == 1:
            block = 'return ' + op + params[0]['name']
        else:
            block = 'return ' + op.join([param['name'] for param in params])
        block += ';'

        if len(params) > 1 and otype not in params[0]['type']:
            if op in _py_op_rev:
                name = _py_op_rev[op]
                params = params[::-1]
            else:
                # do nothing for reflected operator arguments without corresponding python equivalents
                return '', func_inputs
        else:
            name = _py_op[(op, len(params))]
        args = params[1:]
    else:
        raise Exception('No name or operator specified in func')

    comment = obj.get('description', name + ' ...')
    params = ', '.join([param['type'] + ' ' + param['name'] for param in params])
    args = ', '.join([_render_pyarg(arg, templates) for arg in args])
    return _template.format(mod=mod, fname=name,
        func=_render_func(params, args, block, comment)), func_inputs
