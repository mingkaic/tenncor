''' Representation of API files '''

import age.templates.template as template

FILENAME = 'api'

# EXPORT
header = template.AGE_FILE(FILENAME, template.HEADER_EXT,
'''#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{{

{api_decls}

}}

#endif // _GENERATED_API_HPP
''')

def parse_api(api):
    def parse_header_args(arg):
        if 'default' in arg:
            defext = ' = {}'.format(arg['default'])
        else:
            defext = ''
        return '{dtype} {name}{defext}'.format(
            dtype = arg['dtype'],
            name = arg['name'],
            defext = defext)

    if 'description' in api:
        comment = '/**\n{}\n**/\n'.format(
            api['description'])
    else:
        comment = ''
    name = api['name']
    args = ', '.join([parse_header_args(arg) for arg in api['args']])
    return '{comment}ade::TensptrT {api} ({args});'.format(
        comment = comment,
        api = name,
        args = args)

header.api_decls = ('apis', lambda apis: '\n\n'.join([
    parse_api(api) for api in apis]))

# EXPORT
source = template.AGE_FILE(FILENAME, template.SOURCE_EXT,
'''#ifdef _GENERATED_API_HPP

namespace age
{{

{apis}

}}

#endif
''')

def _nullcheck(args):
    tens = list(filter(lambda arg: arg['dtype'] == 'ade::TensptrT', args))
    if len(tens) == 0:
        return 'false'
    varnames = [ten['name'] for ten in tens]
    return ' || '.join([varname + ' == nullptr' for varname in varnames])

source.apis = ('apis', lambda apis: '\n\n'.join(['''ade::TensptrT {api} ({args})
{{
    if ({null_check})
    {{
        logs::fatal("cannot {api} with a null argument");
    }}
    return {retval};
}}'''.format(
    api = api['name'],
    args = ', '.join([arg['dtype'] + ' ' + arg['name']\
        for arg in api['args']]),
    null_check = _nullcheck(api['args']),
    retval = api['out']) for api in apis]))
