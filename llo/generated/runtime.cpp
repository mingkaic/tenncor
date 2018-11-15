#include "llo/generated/code.hpp"
#include "llo/generated/runtime.hpp"

#ifdef _GENERATED_RUNTIME_HPP

namespace age
{

ade::Opcode sum_opcode (void)
{
    return ade::Opcode{"SUM", SUM};
}

ade::Opcode prod_opcode (void)
{
    return ade::Opcode{"PROD", PROD};
}

}

#endif
