from gen.plugin_base import PluginBase

_plugin_id = "OPCODE"

opcode_header = '''
#ifndef _GENERATED_CODES_HPP
#define _GENERATED_CODES_HPP

namespace age
{{

enum _GENERATED_OPCODE
{{
    BAD_OP = 0,
{opcodes}
    _N_GENERATED_OPCODES,
}};

std::string name_op (_GENERATED_OPCODE code);

_GENERATED_OPCODE get_op (std::string name);

template <typename T>
{signature}
{{
    switch (opcode)
    {{
{ops}
        default: logs::fatal("unknown opcode");
    }}{defreturn}
}}

}}

#endif // _GENERATED_CODES_HPP
'''

opcode_source = '''
#ifdef _GENERATED_CODES_HPP

namespace age
{{

struct EnumHash
{{
    template <typename T>
    size_t operator() (T e) const
    {{
        return static_cast<size_t>(e);
    }}
}};

static std::unordered_map<_GENERATED_OPCODE,std::string,EnumHash> code2name =
{{
{code2names}
}};

static std::unordered_map<std::string,_GENERATED_OPCODE> name2code =
{{
{name2codes}
}};

std::string name_op (_GENERATED_OPCODE code)
{{
    auto it = code2name.find(code);
    if (code2name.end() == it)
    {{
        return "BAD_OP";
    }}
    return it->second;
}}

_GENERATED_OPCODE get_op (std::string name)
{{
    auto it = name2code.find(name);
    if (name2code.end() == it)
    {{
        return BAD_OP;
    }}
    return it->second;
}}

}}

#endif
'''

@PluginBase.register
class OpcodesPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        print('processing opcodes')
        print(arguments)
