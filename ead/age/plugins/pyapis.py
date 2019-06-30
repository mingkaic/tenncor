import re
import logging

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

from ead.age.plugins.template import build_template
from ead.age.plugins.apis import api_header

_pybindt = 'PybindT'

_header_template = '''
// type to replace template arguments in pybind
using {pybind} = {pybind_type};
//>>> ^ pybind, pybind_type
'''

_source_template = '''
namespace py = pybind11;

namespace pyage
{{

//>>> unique_wrap
{unique_wrap}

}}

//>>> modname
PYBIND11_MODULE(age, m_{modname})
{{
    m_{modname}.doc() = "pybind for {modname} api";

    //>>> modname
    py::class_<ade::iTensor,ade::TensptrT> tensor(m_{modname}, "Tensor");

    //>>> defs
    {defs}
}}
'''

def _sub_pybind(stmt, source):
    _type_pattern = '([^\w]){}([^\w])'.format(source)
    _type_replace = '\\1{}\\2'.format(_pybindt)
    return re.sub(_type_pattern, _type_replace, ' ' + stmt + ' ').strip()

def _strip_template_prefix(template):
    _template_prefixes = ['typename', 'class']
    for template_prefix in _template_prefixes:
        if template.startswith(template_prefix):
            return template[len(template_prefix):].strip()
    return template

_func_fmt = '''
{outtype} {funcname}_{idx} ({param_decl})
{{
    return {namespace}::{funcname}({args});
}}
'''
def _wrap_func(idx, api, namespace):
    if 'template' in api and len(api['template']) > 0:
        templates = [_strip_template_prefix(typenames)
            for typenames in api['template'].split(',')]
    else:
        templates = []
    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict) and 'type' in api['out']:
        outtype = api['out']['type']

    out = _func_fmt.format(
        outtype=outtype,
        namespace=namespace,
        funcname=api['name'],
        idx=idx,
        param_decl=', '.join([arg['dtype'] + ' ' + arg['name']
            for arg in api['args']]),
        args=', '.join([arg['name'] for arg in api['args']]))
    for temp in templates:
        out = _sub_pybind(out, temp)
    return out

def _handle_pybind(pybind_type):
    return _pybindt

def _handle_pybind_type(pybind_type):
    return pybind_type

def _handle_unique_wrap(pybind_type, apis, namespace):
    return '\n\n'.join([
        _wrap_func(i, api, namespace)
        for i, api in enumerate(apis)]
    )

def _handle_defs(pybind_type, apis, module_name, is_submod):
    _mdef_tmpl = 'm_{module_name}.def("{pyfunc}", '+\
        '&pyage::{func}_{idx}, {description}, {pyargs});'

    _class_def_tmpl = 'py::class_<std::remove_reference<decltype(*{outtype}())>::type,{outtype}>(m_{module_name}, "{name}");'

    cnames = {}
    def checkpy(cname):
        if cname in cnames:
            out = cname + str(cnames[cname])
            cnames[cname] = cnames[cname] + 1
        else:
            out = cname
            cnames[cname] = 0
        return out

    def parse_header_args(arg):
        if 'default' in arg:
            defext = ' = {}'.format(arg['default'])
        else:
            defext = ''
        return '{dtype} {name}{defext}'.format(
            dtype = arg['dtype'],
            name = arg['name'],
            defext = defext)

    def parse_description(arg):
        if 'description' in arg:
            description = ': {}'.format(arg['description'])
        else:
            description = ''
        outtype = 'ade::TensptrT'
        if isinstance(arg['out'], dict) and 'type' in arg['out']:
            outtype = arg['out']['type']
        return '"{outtype} {func} ({args}){description}"'.format(
            outtype = outtype,
            func = arg['name'],
            args = ', '.join([parse_header_args(arg) for arg in arg['args']]),
            description = description)

    def parse_pyargs(arg):
        if 'default' in arg:
            defext = ' = {}'.format(arg['default'])
        else:
            defext = ''
        return 'py::arg("{name}"){defext}'.format(
            name = arg['name'],
            defext = defext)

    outtypes = set()
    for api in apis:
        if 'template' in api and len(api['template']) > 0:
            templates = [_strip_template_prefix(typenames)
                for typenames in api['template'].split(',')]
        else:
            templates = []
        if isinstance(api['out'], dict) and 'type' in api['out']:
            outtype = api['out']['type']
            for temp in templates:
                outtype = _sub_pybind(outtype, temp)
            outtypes.add(outtype)

    class_defs = []
    if not is_submod:
        for outtype in outtypes:
            if 'ade::TensptrT' == outtype:
                continue
            class_defs.append(_class_def_tmpl.format(
                module_name=module_name,
                outtype=outtype,
                name=outtype.split('::')[-1]))

    return '\n    '.join(class_defs) + '\n    ' +\
        '\n    '.join([_mdef_tmpl.format(
            module_name=module_name,
            pyfunc=checkpy(api['name']),
            func=api['name'], idx=i,
            description=parse_description(api),
            pyargs=', '.join([parse_pyargs(arg) for arg in api['args']]))
        for i, api in enumerate(apis)])

_plugin_id = 'PYBINDER'

class PyAPIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _hdr_file = 'pyapi.hpp'
        _submodule_def = '    py::module m_{name} = m_{prename}.def_submodule("{submod}", "A submodule of \'{prename}\'");\n'

        plugin_key = 'api'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        api = arguments[plugin_key]
        bindtype = api.get('pybind_type', 'double')

        generated_files[_hdr_file] = FileRep(
            _header_template.format(
                pybind=_pybindt, pybind_type=bindtype),
                user_includes=[], internal_refs=[])

        contents = {}
        for namespace in api['namespaces']:
            definitions = api['namespaces'][namespace]
            if namespace == '' or namespace == '_':
                module = 'age'
                namespace = ''
            else:
                module = namespace
            uwraps = _handle_unique_wrap(bindtype, definitions, namespace)

            mods = module.split('::')
            mod = mods[0]
            modname = '_'.join(mods)
            mod_def = ''
            is_submod = len(mods) > 1
            if is_submod:
                mod_def = _submodule_def.format(
                    name=modname, prename='_'.join(mods[:-1]), submod=mods[-1])
            defs = mod_def + _handle_defs(bindtype, definitions, modname, is_submod)
            if mod in contents:
                existing_uwraps, existing_defs = contents[mod]
                contents[mod] = (
                    existing_uwraps + '\n\n' + uwraps,
                    existing_defs + '\n\n' + defs)
            else:
                contents[mod] = (uwraps, defs)

        src_file_tmpl = 'pyapi_{}.cpp'
        for mod in contents:
            unique_wrap, defs = contents[mod]
            src_file = src_file_tmpl.format(mod)
            generated_files[src_file] = FileRep(
                _source_template.format(
                    modname=mod,
                    unique_wrap=''.join(unique_wrap),
                    defs=''.join(defs)),
                user_includes=[
                    '"pybind11/pybind11.h"',
                    '"pybind11/stl.h"',
                ],
                internal_refs=[_hdr_file, api_header])

        return generated_files

PluginBase.register(PyAPIsPlugin)
