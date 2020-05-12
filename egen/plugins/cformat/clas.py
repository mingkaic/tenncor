from plugins.template import build_template
from plugins.cformat.args import render as arender
from plugins.cformat.funcs import render_decl as fdecl_render, render_defn as fdefn_render

_template_hdr = '''
{template_decl}struct {class_name}
{{
//>>> init
{init}

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

_template_init = '''{cname} ({args}) {initlist}
{{{do}}}
'''

def _handle_init(obj):
    init = obj.get('init', None)
    if init is not None:
        args = init.get('args', [])
        ilist = init.get('initlist', {})
        doblock = init.get('do', '')

        args = ', '.join([arender(arg) for arg in args])
        if len(ilist) > 0:
            ilist = [
                '{key}({val})'.format(key=k, val=ilist[k])
                for k in ilist
            ]
            ilist = ': ' + ', '.join(ilist)
        else:
            ilist = ''
        if len(doblock) > 0:
            doblock = '\n'.join([''] + [
                '\t' + doline
                for doline in doblock.split('\n')
            ] + [''])
        return _template_init.format(
            cname=obj['name'],
            args=args, initlist=ilist, do = doblock)
    return ''

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

def render(obj, hdr=True):
    if hdr:
        return build_template(_template_hdr, globals(), obj)
    funcs = obj.get('funcs', [])
    return '\n\n'.join([fdefn_render(f, clas=obj) for f in funcs])
