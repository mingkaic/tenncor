from plugins.template import build_template
from plugins.common import strip_template_prefix, get_template_args
from plugins.cformat.args import render as arender
from plugins.cformat.funcs import render_decl as fdecl_render, render_defn as fdefn_render

_template_hdr = '''
{template_decl}struct {class_name}
{{
//>>> init
{init}

//>>> copynmove
{copynmove}

//>>> funcs
{funcs}

//>>> members
{members}
}};
'''

def _handle_template_decl(obj):
    temp = obj.get('template', '').strip()
    if len(temp) > 0:
        return 'template <{}>\n'.format(temp)
    return ''

def _handle_class_name(obj):
    assert('name' in obj)
    return obj['name']

_template_init_decl = '{cname} ({args});'

_template_init_defn = '''{cname}::{cname} ({args}) {initlist}
{{{do}}}
'''

def _handle_init(obj, decl=True):
    if decl:
        temp = _template_init_decl
    else:
        temp = _template_init_defn
    init = obj.get('init', None)
    if init is not None:
        args = init.get('args', [])
        ilist = init.get('initlist', {})
        doblock = init.get('do', '')

        args = ', '.join([arender(arg, accept_def=decl) for arg in args])
        if len(ilist) > 0:
            # order the initializer list by class members declaration
            member = obj.get('members', [])
            mem_order = dict([(mem['name'], i)
                for i, mem in enumerate(member)])
            ikeys = list(ilist.keys())
            assert(all([k in mem_order for k in ikeys]))
            ikeys.sort(key=lambda k: mem_order[k])

            ilist = [
                '{key}({val})'.format(key=k, val=ilist[k])
                for k in ikeys
            ]
            ilist = ': ' + ', '.join(ilist)
        else:
            ilist = ''
        if len(doblock) > 0:
            doblock = '\n'.join([''] + [
                '\t' + doline
                for doline in doblock.split('\n')
            ] + [''])
        return temp.format(
            cname=obj['name'],
            args=args, initlist=ilist, do = doblock)
    return ''

_template_defcopy = '''
{name} (const {typename}& other) = default;
{typename}& operator = (const {typename}& other) = default;
'''

_template_copy_decl = '''
{name} (const {typename}& {oname}{args});
{typename}& operator = (const {typename}& {oname});
'''

_template_copy_defn = '''
{name}::{name} (const {typename}& {oname}{args}){ilist} {{
    {do}
}}
{typename}& {name}::operator = (const {typename}& {oname}) {{
    {do}
    return *this;
}}
'''

_template_defmove = '''
{name} ({typename}&& other) = default;
{typename}& operator = ({typename}&& other) = default;
'''

_template_move_decl = '''
{name} ({typename}&& {oname}{args});
{typename}& operator = ({typename}&& {oname});
'''

_template_move_defn = '''
{name}::{name} ({typename}&& {oname}{args}){ilist} {{
    {do}
}}
{typename}& {name}::operator = ({typename}&& {oname}) {{
    {do}
    return *this;
}}
'''

def _handle_copynmove(obj, decl=True):
    cp = obj.get('copy', None)
    mv = obj.get('move', None)
    if cp is None and mv is None:
        return ''

    name = obj['name']
    templates = [strip_template_prefix(tmp)
        for tmp in get_template_args(obj)]
    if len(templates) > 0:
        typename = name + '<' + ','.join(templates) + '>'
    else:
        typename = name

    if cp is None:
        cp = _template_defcopy.format(name=name, typename=typename)
    else:
        if decl:
            cpytemp = _template_copy_decl
        else:
            cpytemp = _template_copy_defn

        oname = cp.get('other', 'other')
        args = ', '.join([''] + [
            arender(arg, accept_def=decl)
            for arg in cp.get('args', [])
        ])
        ilist = cp.get('initlist', {})
        if len(ilist) > 0:
            ilist = [
                '{key}({val})'.format(key=k, val=ilist[k])
                for k in ilist
            ]
            ilist = ': ' + ', '.join(ilist)
        else:
            ilist = ''

        cp = cpytemp.format(name=name,
            typename=typename, oname=oname,
            args=args, do=cp['do'], ilist=ilist)

    if mv is None:
        mv = _template_defmove.format(name=name, typename=typename)
    else:
        if decl:
            mvtemp = _template_move_decl
        else:
            mvtemp = _template_move_defn

        oname = mv.get('other', 'other')
        args = ', '.join([''] + [
            arender(arg, accept_def=decl)
            for arg in mv.get('args', [])
        ])
        ilist = mv.get('initlist', {})
        if len(ilist) > 0:
            ilist = [
                '{key}({val})'.format(key=k, val=ilist[k])
                for k in ilist
            ]
            ilist = ': ' + ', '.join(ilist)
        else:
            ilist = ''

        mv = mvtemp.format(name=name,
            typename=typename, oname=oname,
            args=args, do=mv['do'], ilist=ilist)

    return cp + '\n\n' + mv

def _handle_funcs(obj):
    funcs = obj.get('funcs', [])
    pubs = []
    privs = []
    for f in funcs:
        if f.get('public', True):
            pubs.append(f)
        else:
            privs.append(f)
    return '\n'.join(
        ['public:'] + [fdecl_render(f) for f in pubs] +
        ['private:'] + [fdecl_render(f) for f in privs])

def _handle_members(obj):
    members = obj.get('members', [])
    pubs = []
    privs = []
    for mem in members:
        if mem.get('public', False):
            pubs.append(mem)
        else:
            privs.append(mem)
    return '\n'.join(
        ['public:'] + [arender(a, True) + ';' for a in pubs] +
        ['private:'] + [arender(a, True) + ';' for a in privs])

def render(api, obj, hdr=True):
    if hdr:
        clas_decls = build_template(_template_hdr, globals(), obj)
        funcs = obj.get('funcs', [])
        if 'template' in obj:
            initfuncs = [
                _handle_init(obj, decl=False),
                _handle_copynmove(obj, decl=False)
            ]
        else:
            initfuncs = []
            funcs = [func for func in funcs if 'template' in func]
        func_decls = '\n\n'.join(initfuncs + [
            fdefn_render(f, root=api, clas=obj)
            for f in funcs
        ])
        return clas_decls, func_decls
    elif 'template' not in obj:
        initfuncs = [
            _handle_init(obj, decl=False),
            _handle_copynmove(obj, decl=False)
        ]
        funcs = obj.get('funcs', [])
        funcs = [func for func in funcs if not 'template' in func]
        func_defns = '\n\n'.join(initfuncs + [
            fdefn_render(f, root=api, clas=obj)
            for f in funcs
        ])
        return '', func_defns
    else:
        return '', ''
