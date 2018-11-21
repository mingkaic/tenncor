''' Representation of gradient mapping files '''

import repr

# EXPORT
header = repr.FILE_REPR("""#ifndef _GENERATED_GRADER_HPP
#define _GENERATED_GRADER_HPP

namespace age
{{

template <typename T>
ade::Tensor* data (T scalar, ade::Shape shape)
{{
    return {scalarize};
}}

struct RuleSet final : public iRuleSet
{{
    ade::Tensor* data (double scalar, ade::Shape shape) override
    {{
        return age::data(scalar, shape);
    }}

    ade::Opcode sum_opcode (void) override
    {{
        return ade::Opcode{{"{sum}", {sum}}};
    }}

    ade::Opcode prod_opcode (void) override
    {{
        return ade::Opcode{{"{prod}", {prod}}};
    }}

    ade::Tensorptr grad_rule (size_t code, TensT args, size_t idx) override;
}};

}}

#endif // _GENERATED_GRADER_HPP
""")

header.scalarize = ("scalarize", lambda scalarize: scalarize)

header.sum = ("sum", lambda sum: sum)

header.prod = ("prod", lambda prod: prod)

# EXPORT
source = repr.FILE_REPR("""#ifdef _GENERATED_GRADER_HPP

namespace age
{{

std::shared_ptr<iRuleSet> Grader::default_rules = std::make_shared<RuleSet>();

ade::Tensorptr RuleSet::grad_rule (size_t code,TensT args,size_t idx)
{{
    switch (code)
    {{
{gradops}
        default: err::fatal("no gradient rule for unknown opcode");
    }}
}}

}}

#endif
""")

source.gradops = ("grads", lambda gradmap: '\n'.join(["        case {code}: return {retval};".format(\
    code = code, retval = gradmap[code]) for code in gradmap]))