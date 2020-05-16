from plugins.pformat.funcs import render_classfunc, pybindt, clean_templates

_template = 'py::class_<{cname}> cls_{name}({mod}, "{name}");'

def _render_classmems(obj, name, cname):
    mem = obj['name']
    return 'cls_{name}.def_readwrite("{mem}", &{cname}::{mem});'.format(
        name=name, cname=cname, mem=mem)

def render(obj, mod, namespace):
    templates = obj.get('template', '').strip().split(',')
    ntemplates = len(templates)
    name = obj['name']
    cname = namespace + '::' + name
    if ntemplates > 0:
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

    if 'init' in obj:
        init_args = obj['init'].get('args', [])
        init_args = [arg['pyreplace'] if 'pyreplace' in arg else arg for arg in init_args]
        for arg in init_args:
            arg['type'] = clean_templates(arg['type'], templates)

        params = ', '.join([arg['type'] + ' ' + arg['name'] for arg in init_args])
        pnames = ', '.join([arg['convert'] if 'convert' in arg else arg['name'] for arg in init_args])
        args = ', '.join([''] + ['py::arg("{}")={}'.format(arg['name'], str(arg['default']))
            if 'default' in arg else 'py::arg("{}")'.format(arg['name']) for arg in init_args])
        func_defs.append('cls_{name}.def(py::init([]({params}){{ return {cname}({pnames}); }}){args});'.format(
            name=name, cname=cname, params=params, pnames=pnames, args=args))

    return '\n\n'.join([_template.format(cname=cname, name=name, mod=mod)] + func_defs +
        [_render_classmems(mem, name, cname) for mem in mems if mem.get('public', False)]), class_inputs
