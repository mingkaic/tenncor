//
//  functor.cpp
//  wire
//

#include "wire/functor.hpp"

#ifdef WIRE_FUNCTOR_HPP

namespace wire
{

static std::vector<std::string> to_ids (std::vector<Identifier*> ids)
{
    std::vector<std::string> out(ids.size());
    std::transform(ids.begin(), ids.end(), out.begin(),
    [](Identifier* id) -> std::string
    {
        return id->get_uid();
    });
    return out;
}

Functor::Functor (std::vector<Identifier*> args,
    slip::OPCODE opcode, Graph& graph) :
    Identifier(&graph,
    new mold::Functor(
        to_nodes(args),
        slip::forward_op(opcode),
        slip::backward_op(opcode)),
    slip::opnames[opcode]),
    arg_ids_(to_ids(args)) {}

std::vector<mold::iNode*> Functor::to_nodes (std::vector<Identifier*> ids)
{
    std::vector<mold::iNode*> out(ids.size());
    std::transform(ids.begin(), ids.end(), out.begin(),
    [](Identifier* id) -> mold::iNode*
    {
        return id->arg_.get();
    });
    return out;
}

}

#endif
