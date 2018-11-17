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

header.api_decls = ("apis", lambda apis: '\n\n'.join(["ade::Tensorptr {api} ({args});".format(\
    api = api["name"], args = ', '.join(api["args"])) for api in apis]))

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_API_HPP

namespace age
{{

{apis}

}}

#endif
""")

source.apis = ("apis", lambda apis: '\n\n'.join(["""ade::Tensorptr {api} ({args})
{{
    return {retval};
}}""".format(api = api["name"], args = ', '.join(api["args"]), retval = api["out"]) for api in apis]))
