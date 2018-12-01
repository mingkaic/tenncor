''' Representation of C API files '''

import repr

_origtype = 'ade::TensptrT'
_repltype = 'int64_t'

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_CAPI_HPP
#define _GENERATED_CAPI_HPP

int64_t malloc_tens (void* ptr);

void* get_ptr (int64_t id);

extern void free_tens (int64_t id);

extern void get_shape (int outshape[8], int64_t tens);

{api_decls}

#endif // _GENERATED_CAPI_HPP
""")

header.api_decls = ("apis", lambda apis: '\n\n'.join([\
    "extern int64_t age_{func} ({args});".format(\
    func = api["name"], args = ', '.join([\
        arg.replace(_origtype, _repltype)\
        for arg in api["args"]])) for api in apis]))

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_CAPI_HPP

static std::unordered_map<int64_t,ade::TensptrT> tens;

inline ade::TensptrT get_tens (int64_t id)
{{
    auto it = tens.find(id);
    if (tens.end() == it)
    {{
        return ade::TensptrT(nullptr);
    }}
    return it->second;
}}

int64_t malloc_tens (void* ptr)
{{
    int64_t id = (int64_t) ptr;
    tens.emplace(id, ade::TensptrT(static_cast<ade::iTensor*>(ptr)));
    return id;
}}

void* get_ptr (int64_t id)
{{
    return get_tens(id).get();
}}

void free_tens (int64_t id)
{{
    tens.erase(id);
}}

void get_shape (int outshape[8], int64_t id)
{{
    const ade::Shape& shape = get_tens(id)->shape();
    std::copy(shape.begin(), shape.end(), outshape);
}}

{apis}

#endif
""")

_cfunc_fmt = """int64_t age_{func} ({params})
{{
    {arg_decls}auto ptr = age::{func}({retargs});
    int64_t id = (int64_t) ptr.get();
    tens.emplace(id, ptr);
    return id;
}}"""

def _defn_func(api):
    func = api["name"]
    vars = [arg.split(' ') for arg in api["args"]]
    typevars = [(var[0], var[-1]) for var in vars]
    params = []
    arg_decls = []
    args = []
    for typevar in typevars:
        if typevar[0] == _origtype:
            params.append('int64_t {}'.format(typevar[1]))
            arg_decls.append('ade::TensptrT {name}_ptr = get_tens({name});'
                .format(name=typevar[1]))
            args.append(typevar[1] + '_ptr')
        else:
            params.append(' '.join(typevar))
            args.append(typevar[1])
    arg_decls_str = '\n    '.join(arg_decls)
    if len(arg_decls) > 0:
        arg_decls_str = arg_decls_str + '\n    '
    return _cfunc_fmt.format(
        func = func,
        params = ', '.join(params),
        arg_decls = arg_decls_str,
        retargs = ', '.join(args))

source.apis = ("apis", lambda apis: '\n\n'.join([_defn_func(api) for api in apis]))
