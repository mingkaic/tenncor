''' Representation of API files '''

import repr

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_API_HPP
#define _GENERATED_API_HPP

namespace age
{{

{api_decls}

}}

#endif // _GENERATED_API_HPP
""")

header.api_decls = ("apis", lambda apis: '\n\n'.join(["ade::TensptrT {api} ({args});".format(\
    api = api["name"], args = ', '.join(api["args"])) for api in apis]))

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_API_HPP

namespace age
{{

{apis}

}}

#endif
""")

def _nullcheck(args):
    vars = [arg.split(' ') for arg in args]
    tens = list(filter(lambda vpair: vpair[0] == 'ade::TensptrT',\
        [(var[0], var[-1]) for var in vars]))
    if len(tens) == 0:
        return "false"
    varnames = [ten[-1] for ten in tens]
    return " || ".join([varname + " == nullptr" for varname in varnames])

source.apis = ("apis", lambda apis: '\n\n'.join(["""ade::TensptrT {api} ({args})
{{
    if ({null_check})
    {{
        logs::fatal("cannot {api} with a null argument");
    }}
    return {retval};
}}""".format(
    api = api["name"],
    args = ', '.join(api["args"]),
    null_check = _nullcheck(api["args"]),
    retval = api["out"]) for api in apis]))
