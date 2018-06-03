//
//  functor.cpp
//  wire
//

#include <algorithm>

#include "wire/functor.hpp"
#include "wire/constant.hpp"
#include "wire/error.hpp"

#ifdef WIRE_FUNCTOR_HPP

namespace wire
{

static std::vector<std::string> to_ids (std::vector<Identifier*> ids)
{
	std::vector<std::string> out(ids.size());
	std::transform(ids.begin(), ids.end(), out.begin(),
	[](Identifier * id) -> std::string
	{
		return id->get_uid();
	});
	return out;
}

Functor::Functor (std::vector<Identifier*> args,
	slip::OPCODE opcode, GradF grad, Graph& graph) :
	Identifier(&graph,
		new mold::Functor(
			to_nodes(args),
			slip::get_op(opcode)),
		slip::opnames[opcode]),
	arg_ids_(to_ids(args)),
	grad_(grad)
{
	for (Identifier* arg : args)
	{
		// validate
		std::string uid = arg->get_uid();
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

Identifier* Functor::derive (Identifier* wrt)
{
	if (false == get()->has_data())
	{
		throw mold::UninitializedError();
	}
	Identifier* out;
	if (this == wrt)
	{
		out = make_one(this);
	}
	else
	{
		std::vector<Identifier*> args(arg_ids_.size());
		std::transform(arg_ids_.begin(), arg_ids_.end(), args.begin(),
		[this](std::string id)
		{
			return graph_->get_node(id);
		});
		out = grad_(wrt, args);
	}
	return out;
}

std::vector<mold::iNode*> Functor::to_nodes (std::vector<Identifier*> ids)
{
	std::vector<mold::iNode*> out(ids.size());
	std::transform(ids.begin(), ids.end(), out.begin(),
	[](Identifier * id) -> mold::iNode*
	{
		return id->get();
	});
	return out;
}

}

#endif
