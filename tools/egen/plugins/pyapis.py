import re
import logging

from tools.gen.plugin_base import PluginBase
from tools.gen.file_rep import FileRep

from plugins.template import build_template
from plugins.apis import api_header
from plugins.common import order_classes, reference_classes
from plugins.cformat.funcs import render_defn as cfrender
from plugins.pformat.clas import render as crender
from plugins.pformat.funcs import render as frender, process_modname, pybindt

_header_template = '''
// type to replace template arguments in pybind
using {pybind} = {pybind_type};
//>>> ^ pybind, pybind_type
'''

_source_template = '''
namespace py = pybind11;

//>>> global_decs
{global_decs}

//>>> modname
PYBIND11_MODULE({modname}, m_{modname})
{{
m_{modname}.doc() = "pybind for {modname} api";

//>>> input_defs
{input_defs}

#ifdef CUSTOM_PYBIND_EXT
//>>> modname
CUSTOM_PYBIND_EXT(m_{modname})
#endif

//>>> content
{content}

}}
'''

_content_template = '''
//>>> class_defs
{class_defs}

//>>> global_defs
{global_defs}

//>>> func_defs
{func_defs}
'''

def render_pyglobal_decl(mems):
    out = []
    for mem in mems:
        assert('name' in mem and 'type' in mem)
        affix = ''
        if 'decl' in mem:
            out.append(cfrender({
                'name': 'py_' + mem['name'] + '_global',
                'out': {
                    'type': mem['type'] + '&',
                    'val': mem['decl']
                }
            }))
        else:
            if 'val' in mem:
                affix = '= ' + mem['val']
            out.append(' '.join([mem['type'], mem['name'], affix + ';']))
    return out

def render_pyclasses(classes, ext_path, mod, namespace):
    classes = reference_classes(classes, ext_path)
    class_defs = dict()
    class_inputs = dict()
    for clas in classes:
        class_def, cinputs = crender(clas, mod, namespace)
        cname = clas['name']
        class_defs[cname] = class_def
        class_inputs.update(cinputs)

    order = order_classes(classes)
    return [class_defs[clas] for clas in order], class_inputs

def render_pyglobal(mem, mod):
    global_tmpl = '{mod}.attr("{name}") = &{val};'
    name = mem['name']
    if 'decl' in mem:
        val = 'py_' + name + '_global()'
    else:
        val = name
    return global_tmpl.format(mod=mod, name=name, val=val)

def render_pyapi(api, ext_path, mod, ns=''):
    _submodule_def = 'py::module m_{submod} = {mod}.def_submodule("{submod}", "A submodule of \'{mod}\'");'

    global_decls = []
    content_lines = []
    input_types = dict()

    if 'namespaces' in api:
        for ns in api['namespaces']:
            namespace = ns['name']
            sub_decls, sub_content, sub_inputs = render_pyapi(
                ns['content'], ext_path, 'm_' + namespace, ns + '::' + namespace)

            global_decls += sub_decls
            content_lines += [_submodule_def.format(submod=namespace, mod=mod)] + sub_content
            input_types.update(sub_inputs)

    global_mems = api.get('pyglobal', [])
    funcs = api.get('funcs', [])
    classes = api.get('classes', [])

    funcs = list(filter(lambda f: not f.get('pyignores', False), funcs))

    global_decls += render_pyglobal_decl(global_mems)

    class_content, class_inputs = render_pyclasses(classes, ext_path, mod, ns)
    content_lines += class_content
    input_types.update(class_inputs)

    content_lines += [render_pyglobal(mem, mod) for mem in global_mems]
    for f in funcs:
        fcontent, finputs = frender(f, mod, ns)
        content_lines.append(fcontent)
        input_types.update(finputs)

    return global_decls, content_lines, input_types

_plugin_id = 'PYBINDER'

_pyapi_header = 'pyapi.hpp'

@PluginBase.register
class PyAPIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments, **kwargs):
        plugin_key = 'api'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        api = arguments[plugin_key]

        bindtype = api.get('pybind_type', 'double')

        generated_files[_pyapi_header] = FileRep(
            _header_template.format(pybind=pybindt, pybind_type=bindtype),
                user_includes=['"eteq/etens.hpp"'], internal_refs=[])

        # split modules by top-level namespaces
        modname = api['pybind_module']
        ignore_types = [process_modname(dtype)
            for dtype in api.get('pyignore_type', [])] + [process_modname(pybindt)]

        assert('ext_path' in kwargs)
        ext_path = kwargs['ext_path']

        decls, content_lines, input_types = render_pyapi(api, ext_path, 'm_' + modname)
        src_file = 'pyapi_{}.cpp'.format(modname)
        generated_files[src_file] = FileRep(
            _source_template.format(
                modname=modname,
                input_defs='\n'.join([input_types[input_mod]
                    for input_mod in input_types if input_mod not in ignore_types]),
                global_decs='\n'.join(decls),
                content='\n\n'.join(content_lines)),
            user_includes=[
                '"pybind11/pybind11.h"',
                '"pybind11/stl.h"',
                '"pybind11/operators.h"',
                '"internal/global/config.hpp"',
                '"internal/eigen/device.hpp"',
            ] + api.get('pybind_includes', []),
            internal_refs=[_pyapi_header, api_header])

        return generated_files
