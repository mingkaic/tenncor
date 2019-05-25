import logging

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

from ead.generator.plugins.template import build_template

_header_template = '''
#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{{

//>>> api_header
{api_header}

}}

#endif // _GENERATED_API_HPP
'''

_source_template = '''
#ifdef _GENERATED_API_HPP

namespace age
{{

//>>> api_source
{api_source}

}}

#endif
'''

def _parse_args(arg, accept_def = True):
    if 'default' in arg and accept_def:
        defext = ' = {}'.format(arg['default'])
    else:
        defext = ''
    return '{dtype} {name}{defext}'.format(
        dtype = arg['dtype'],
        name = arg['name'],
        defext = defext)

def _nullcheck(args):
    tens = list(filter(lambda arg: arg['dtype'] == 'ade::TensptrT', args))
    if len(tens) == 0:
        return 'false'
    varnames = [ten['name'] for ten in tens]
    return ' || '.join([varname + ' == nullptr' for varname in varnames])

_decl_tmp = '''
/// {comment}
{outtype} {funcname} ({args});
'''
def _decl_func(api):
    funcname = api['name']
    comment = api.get('description', funcname + ' ...')

    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict) and 'type' in api['out']:
        outtype = api['out']['type']

    return _decl_tmp.format(
        comment = comment,
        outtype = outtype,
        funcname = funcname,
        args = ', '.join([
            _parse_args(arg)
            for arg in api['args']
        ]))

_defn_tmp = '''
/// {comment}
{template_prefix}{outtype} {funcname} ({args})
{{
    if ({null_check})
    {{
        logs::fatal("cannot {funcname} with a null argument");
    }}
    {block}
}}
'''
def _defn_func(api):
    funcname = api['name']
    comment = api.get('description', funcname + ' ...')
    template = api.get('template', '')

    # treat as if header
    if len(template) > 0:
        template_prefix = 'template <{}>\n'.format(template)
        args = [(arg, True) for arg in api['args']]
    else:
        template_prefix = ''
        args = [(arg, False) for arg in api['args']]

    outtype = 'ade::TensptrT'
    if isinstance(api['out'], dict):
        if 'type' in api['out']:
            outtype = api['out']['type']
        outval = api['out']['val']
    else:
        outval = api['out']

    return _defn_tmp.format(
        comment = comment,
        template_prefix = template_prefix,
        outtype = outtype,
        funcname = funcname,
        args = ', '.join([
            _parse_args(*arg)
            for arg in args
        ]),
        null_check = _nullcheck(api['args']),
        block = outval)

def _handle_api_header(apis):
    return '\n\n'.join([
        _defn_func(api) if 'template' in api and len(api['template']) > 0
        else _decl_func(api)
        for api in apis
    ])

def _handle_api_source(apis):
    return '\n\n'.join([
        _defn_func(api) for api in apis
        if 'template' not in api or len(api['template']) == 0
    ])

_plugin_id = "API"

api_header = 'api.hpp'

class APIsPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        _src_file = 'api.cpp'
        plugin_key = 'api'
        if plugin_key not in arguments:
            logging.warning(
                'no relevant arguments found for plugin %s', _plugin_id)
            return

        module = globals()
        api = arguments[plugin_key]

        generated_files[api_header] = FileRep(
            build_template(_header_template, module, api['definitions']),
            user_includes=api.get('includes', []),
            internal_refs=[])

        generated_files[_src_file] = FileRep(
            build_template(_source_template, module, api['definitions']),
            user_includes=[],
            internal_refs=[api_header])

        return generated_files

PluginBase.register(APIsPlugin)
