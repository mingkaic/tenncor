//
//  registry.cpp
//  slip
//

#include "slip/registry.hpp"

#ifdef SLIP_REGISTRY_HPP

namespace slip
{

struct OperaBundle
{
    mold::ArgsF fwd_;
    mold::GradF bwd_;
    mold::ShaperF shaper_;
    mold::TyperF typer_;
};

std::unordered_map<OPCODE,OperaBundle> registry =
{
    
};

mold::OperateIO forward_op (OPCODE opcode)
{
    // auto op = registry[opcode];
    // return mold::OperateIO(op.fwd_, op.shape_, op.type_);
    return mold::OperateIO([](clay::State&,std::vector<clay::State>) {},
        [](std::vector<clay::Shape>)->clay::Shape{ return clay::Shape(); },
        [](std::vector<clay::DTYPE>)->clay::DTYPE{ return clay::DTYPE::BAD; });
}

mold::GradF backward_op (OPCODE opcode)
{
    // auto op = registry[opcode];
    // return op.bwd_;
    return [](mold::iNode*, std::vector<mold::iNode*>) -> mold::iNode*
    {
        return nullptr;
    };
}

}

#endif
