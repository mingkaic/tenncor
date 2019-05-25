import logging

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

from ead.generator.plugins.template import build_template
from ead.generator.plugins.apis import api_header

_PYBINDT = '<PybindT>'

_header_template = '''
// type to replace template arguments in pybind
using PybindT = {pybind_type};
//>>> ^ unique_wrap
'''

_source_template = '''
namespace py = pybind11;

namespace pyage
{{

//>>> unique_wrap
{unique_wrap}

}}

PYBIND11_MODULE(age, m)
{{
    m.doc() = "pybind ade generator";

    py::class_<ade::iTensor,ade::TensptrT> tensor(m, "Tensor");

    //>>> defs
    {defs}
}}
'''

def _strip_template_prefix(template):
    _template_prefixes = ['typename', 'class']
    for template_prefix in _template_prefixes:
        if template.startswith(template_prefix):
            return template[len(template_prefix):].strip()
    return template

_func_fmt = '''
{outtype} {funcname}_{idx} ({param_decl})
{{
    return age::{funcname}({args});
}}
'''
def _wrap_func(idx, api):
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
        funcname = api['name'],
        idx = idx,
        param_decl = ', '.join([arg['dtype'] + ' ' + arg['name']
            for arg in api['args']]),
        args = ', '.join([arg['name'] for arg in api['args']]))
    for temp in templates:
        out = out.replace('<{}>'.format(temp), _PYBINDT)
    return out

def _handle_pybind_type(pybind_type, apis):
    return pybind_type

def _handle_unique_wrap(pybind_type, apis):
    return '\n\n'.join([
        _wrap_func(i, api)
        for i, api in enumerate(apis)]
    )

_mdef_fmt = 'm.def("{pyfunc}", &pyage::{func}_{idx}, {description}, {pyargs});'
def _handle_defs(pybind_type, apis):
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
                outtype = outtype.replace('<{}>'.format(temp), _PYBINDT)
            outtypes.add(outtype)

    class_defs = []
    for outtype in outtypes:
        if 'ade::TensptrT' == outtype:
            continue
        class_defs.append('py::class_<std::remove_reference<decltype(*{outtype}())>::type,{outtype}>(m, "{name}");'.format(
            outtype=outtype,
            name=outtype.split('::')[-1]))

    return '\n    '.join(class_defs) + '\n    ' +\
        '\n    '.join([_mdef_fmt.format(
        pyfunc = checkpy(api['name']),
        func = api['name'],
        idx = i,
        description = parse_description(api),
        pyargs = ', '.join([parse_pyargs(arg) for arg in api['args']]))
        for i, api in enumerate(apis)])

_plugin_id = 'PYBINDER'

class PyAPIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _hdr_file = 'pyapi.hpp'
        _src_file = 'pyapi.cpp'
        plugin_key = 'api'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        module = globals()
        api = arguments[plugin_key]
        bindtype = api.get('pybind_type', 'double')

        generated_files[_hdr_file] = FileRep(
            build_template(_header_template, module,
                bindtype, api['definitions']),
            user_includes=[],
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            build_template(_source_template, module,
                bindtype, api['definitions']),
            user_includes=[
                '"pybind11/pybind11.h"',
                '"pybind11/stl.h"',
            ],
            internal_refs=[_hdr_file, api_header])

        return generated_files

PluginBase.register(PyAPIsPlugin)
