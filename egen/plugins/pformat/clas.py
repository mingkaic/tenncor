from plugins.pformat.funcs import render_classfunc, pybindt

_template = 'py::class_<{cname}> cls_{name}({mod}, "{name}");'

def _render_classmems(obj, name, cname):
    mem = obj['name']
    return 'cls_{name}.def_readwrite("{mem}", &{cname}::{mem});'.format(
        name=name, cname=cname, mem=mem)

def render(obj, mod, namespace):
    ntemplates = len(obj.get('template', '').split(','))
    name = obj['name']
    temp = obj.get('template', '').strip()
    cname = namespace + '::' + name
    if len(temp) > 0:
        cname += '<' + ','.join([pybindt] * ntemplates) + '>'
    funcs = obj.get('funcs', [])
    mems = obj.get('members', [])

    func_defs = []
    class_inputs = dict()
    for f in funcs:
        if f.get('public', True):
            fdef, finputs = render_classfunc(f, obj, namespace, mod)
            func_defs.append(fdef)
            class_inputs.update(finputs)

    return '\n\n'.join([_template.format(cname=cname, name=name, mod=mod)] + func_defs +
        [_render_classmems(mem, name, cname) for mem in mems if mem.get('public', False)]), class_inputs
