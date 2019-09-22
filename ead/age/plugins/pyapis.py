import re
import logging

from gen.plugin_base import PluginBase
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

//>>> name_mangling
{name_mangling}

}}

//>>> modname
PYBIND11_MODULE({modname}, m_{modname})
{{
    m_{modname}.doc() = "pybind for {modname} api";

    //>>> modname
    py::class_<ade::iTensor,ade::TensptrT> tensor(m_{modname}, "Tensor");

    //>>> defs
    {defs}
}}
'''

def _sub_pybind(stmt, source):
    _type_pattern = '([^\\w]){}([^\\w])'.format(source)
    _type_replace = '\\1{}\\2'.format(_pybindt)
    return re.sub(_type_pattern, _type_replace, ' ' + stmt + ' ').strip()

def _strip_template_prefix(template):
    _template_prefixes = ['typename', 'class']
    for template_prefix in _template_prefixes:
        if template.startswith(template_prefix):
            return template[len(template_prefix):].strip()
    # todo: parse valued templates variable (e.g.: size_t N)
    return template

_func_fmt = '''
{outtype} {funcname}_{idx} ({param_decl})
{{
    return {namespace}::{funcname}({args});
}}
'''
def _mangle_func(idx, api, namespace):
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

    for typenames in api.get('template', '').split(','):
        out = _sub_pybind(out, _strip_template_prefix(typenames))
    return out

def _handle_pybind(pybind_type):
    return _pybindt

def _handle_pybind_type(pybind_type):
    return pybind_type

def _handle_name_mangling(pybind_type, apis, namespace):
    return '\n\n'.join([
        _mangle_func(i, api, namespace)
        for i, api in enumerate(apis)]
    )

def _parse_header_args(arg):
    if 'default' in arg:
        defext = ' = {}'.format(arg['default'])
    else:
        defext = ''
    return '{dtype} {name}{defext}'.format(
        dtype = arg['dtype'],
        name = arg['name'],
        defext = defext)

def _parse_description(arg):
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
        args = ', '.join([_parse_header_args(arg) for arg in arg['args']]),
        description = description)

def _parse_pyargs(arg):
    if 'default' in arg:
        defext = ' = {}'.format(arg['default'])
    else:
        defext = ''
    return 'py::arg("{name}"){defext}'.format(
        name = arg['name'],
        defext = defext)

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

__py_op_rev = {
    '+': '__radd__',
    '*': '__rmul__',
    '-': '__rsub__',
    '/': '__rtruediv__',
}

_def_op_tmpl = '{label}.def("{pyop}", []({params}){{ return {operator}; }}, py::is_operator());'

def _def_op(t2labels, api):
    templates = [_strip_template_prefix(typenames)
        for typenames in api.get('template', '').split(',')]

    rep_type = 'ade::TensptrT'
    label = 'tensor'
    if isinstance(api['out'], dict) and 'type' in api['out']:
        rep_type = api['out']['type']
        label_type = rep_type
        for template in templates:
            label_type = _sub_pybind(label_type, template)
        label = t2labels.get(label_type, label)

    op = api['operator']
    args = [arg['name'] for arg in api['args']]
    if len(args) == 1:
        operator = op + args[0]
    else:
        operator = op.join(args)

    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict) and 'type' in api['out']:
        outtype = api['out']['type']

    params = [arg['dtype'] + ' ' + arg['name'] for arg in api['args']]
    if len(api['args']) > 1 and\
        api['args'][0]['dtype'] != outtype and op in __py_op_rev:
        pyop = __py_op_rev[op]
        params = params[::-1]
    else:
        pyop = _py_op[(op, len(api['args']))]

    out = _def_op_tmpl.format(
        label = label,
        pyop = pyop,
        params = ', '.join(params),
        operator = operator,
    )

    for typenames in api.get('template', '').split(','):
        out = _sub_pybind(out, _strip_template_prefix(typenames))
    return out

def _handle_defs(pybind_type, apis, module_name, first_module):
    _mdef_tmpl = 'm_{module_name}.def("{func}", '+\
        '&pyage::{func}_{idx}, {description}, {pyargs});'

    _class_def_tmpl = 'py::class_<std::remove_reference<decltype(*{outtype}())>::type,{outtype}> {label}(m_{module_name}, "{name}");'

    outtypes = set()
    for api in apis:
        templates = [_strip_template_prefix(typenames)
            for typenames in api.get('template', '').split(',')]
        if isinstance(api['out'], dict) and 'type' in api['out']:
            outtype = api['out']['type']
            for temp in templates:
                outtype = _sub_pybind(outtype, temp)
            outtypes.add(outtype)

    class_defs = []
    atype_labels = {}
    if first_module:
        for i, outtype in enumerate(outtypes):
            if 'ade::TensptrT' == outtype:
                continue
            label = 'class_{}'.format(i)
            atype_labels[outtype] = label
            class_defs.append(_class_def_tmpl.format(
                module_name=module_name,
                outtype=outtype,
                label=label,
                name=outtype.split('::')[-1]))

    func_defs = [_mdef_tmpl.format(
            module_name=module_name,
            func=api['name'], idx=i,
            description=_parse_description(api),
            pyargs=', '.join([_parse_pyargs(arg) for arg in api['args']]))
        for i, api in enumerate(apis)]

    operator_defs = [_def_op(atype_labels, api) for api in apis if 'operator' in api]

    defs = [
        '\n    '.join(class_defs),
        '\n\n    '.join(func_defs),
        '\n\n    '.join(operator_defs),
    ]
    return '\n\n    '.join([d for d in defs if len(d) > 0])

_plugin_id = 'PYBINDER'

class PyAPIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _hdr_file = 'pyapi.hpp'
        _submodule_def = '    py::module m_{name} = m_{prename}.def_submodule("{submod}", "A submodule of \'{prename}\'");\n    '

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
            uwraps = _handle_name_mangling(bindtype, definitions, namespace)

            mods = module.split('::')
            mod = mods[0]
            modname = '_'.join(mods)
            mod_def = ''
            if len(mods) > 1:
                mod_def = _submodule_def.format(
                    name=modname, prename='_'.join(mods[:-1]), submod=mods[-1])
            defs = mod_def + _handle_defs(bindtype, definitions, modname, mod not in contents)
            if mod in contents:
                existing_uwraps, existing_defs = contents[mod]
                contents[mod] = (
                    existing_uwraps + '\n\n' + uwraps,
                    existing_defs + '\n\n' + defs)
            else:
                contents[mod] = (uwraps, defs)

        src_file_tmpl = 'pyapi_{}.cpp'
        for mod in contents:
            name_mangling, defs = contents[mod]
            src_file = src_file_tmpl.format(mod)
            generated_files[src_file] = FileRep(
                _source_template.format(
                    modname=mod,
                    name_mangling=''.join(name_mangling),
                    defs=''.join(defs)),
                user_includes=[
                    '"pybind11/pybind11.h"',
                    '"pybind11/stl.h"',
                    '"pybind11/operators.h"',
                ],
                internal_refs=[_hdr_file, api_header])

        return generated_files

PluginBase.register(PyAPIsPlugin)
