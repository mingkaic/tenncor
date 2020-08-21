import logging

from tools.gen.plugin_base import PluginBase
from tools.gen.file_rep import FileRep

from plugins.template import build_template
from plugins.common import order_classes, reference_classes
from plugins.cformat.clas import render as crender, _handle_template_decl, _handle_class_name
from plugins.cformat.funcs import render_decl as fdecl, render_defn as fdefn

_header_template = '''
#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

//>>> content
{content}

#endif // _GENERATED_API_HPP
'''

_source_template = '''
#ifdef _GENERATED_API_HPP

//>>> content
{content}

#endif
'''

def predef_class(obj):
    return build_template('{template_decl}struct {class_name};', {
        '_handle_template_decl': _handle_template_decl,
        '_handle_class_name': _handle_class_name}, obj)

def render_classes(classes, defn_funcs=True):
    if defn_funcs:
        class_defs = dict()
        for clas in classes:
            class_def = crender(clas, True)
            cname = clas['name']
            class_defs[cname] = class_def

        order = order_classes(classes)
        return [predef_class(clas) for clas in classes] +\
            [class_defs[clas] for clas in order]
    return [crender(clas, False) for clas in classes]

def render_globals(mems, hdr):
    out = []
    for mem in mems:
        assert('name' in mem and 'type' in mem)
        prefix = ''
        affix = ''
        if hdr:
            prefix = 'extern'
        elif 'val' in mem:
            affix = '= ' + mem['val']
        out.append(' '.join([
            prefix, mem['type'], mem['name'], affix + ';']))
    return out

def render_api(api, ext_path, hdr):
    lines = []
    if 'namespaces' in api:
        for ns in api['namespaces']:
            namespace = ns['name']
            lines += ['namespace ' + namespace + '{', ''] +\
                render_api(ns['content'], ext_path, hdr) + ['', '}', '']

    global_mems = api.get('global', [])
    funcs = api.get('funcs', [])
    classes = api.get('classes', [])
    classes = reference_classes(classes, ext_path)

    temp_funcs = [f for f in funcs if len(f.get('template', '')) > 0]
    norm_funcs = [f for f in funcs if len(f.get('template', '')) == 0]

    temp_class = [c for c in classes if len(c.get('template', '')) > 0]
    norm_class = [c for c in classes if len(c.get('template', '')) == 0]

    if hdr:
        lines += render_classes(classes, defn_funcs=True)
        lines += render_globals(global_mems, hdr)
        lines += [fdecl(f) for f in funcs]

        lines += render_classes(temp_class, defn_funcs=False)
        lines += [fdefn(f) for f in temp_funcs]
    else:
        lines += render_globals(global_mems, hdr)
        lines += render_classes(norm_class, defn_funcs=False)
        lines += [fdefn(f) for f in norm_funcs]
    return lines

_plugin_id = "API"

api_header = 'api.hpp'

@PluginBase.register
class APIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments, **kwargs):
        _src_file = 'api.cpp'
        plugin_key = 'api'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        assert('ext_path' in kwargs)
        ext_path = kwargs['ext_path']

        api = arguments[plugin_key]

        hdr_lines = render_api(api, ext_path, hdr=True)
        src_lines = render_api(api, ext_path, hdr=False)

        generated_files[api_header] = FileRep(
            _header_template.format(content='\n\n'.join(hdr_lines)),
            user_includes=api.get('includes', []),
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            _source_template.format(content='\n\n'.join(src_lines)),
            user_includes=[],
            internal_refs=[api_header])

        return generated_files
