//
//  functor.cpp
//  wire
//

#include <algorithm>

#include "wire/functor.hpp"
#include "wire/operators.hpp"
#include "wire/error.hpp"

#ifdef WIRE_FUNCTOR_HPP

namespace wire
{

static std::vector<mold::iNode*> to_nodes (std::vector<Identifier*> ids)
{
	std::vector<mold::iNode*> out(ids.size());
	std::transform(ids.begin(), ids.end(), out.begin(),
	[](Identifier* id) -> mold::iNode*
	{
		return id->get();
	});
	return out;
}

static std::vector<UID> to_ids (std::vector<Identifier*> ids)
{
	std::vector<UID> out(ids.size());
	std::transform(ids.begin(), ids.end(), out.begin(),
	[](Identifier * id) -> UID
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
			slip::get_op(opcode)),
		slip::opnames.at(opcode)),
	opcode_(opcode), arg_ids_(to_ids(args))
{
	for (Identifier* arg : args)
	{
		// validate
		UID uid = arg->get_uid();
		if (false == graph.has_node(uid))
		{
			throw MissingNodeError(graph.get_gid(), uid);
		}
	}
	graph_->add_func(opcode, this);
}

Functor::~Functor (void)
{
	graph_->remove_func(this);
}

}

#endif
