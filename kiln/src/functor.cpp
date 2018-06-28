//
//  functor.cpp
//  kiln
//

#include <algorithm>

#include "kiln/functor.hpp"
#include "kiln/operators.hpp"
#include "kiln/error.hpp"

#ifdef KILN_FUNCTOR_HPP

namespace kiln
{

Functor::Functor (std::vector<Identifier*> args,
	slip::OPCODE opcode, Graph& graph) :
	Identifier(&graph,
		new mold::Functor(
			[](std::vector<Identifier*> args) -> std::vector<mold::DimRange>
			{
				std::vector<mold::DimRange> out;
				std::transform(args.begin(), args.end(), std::back_inserter(out),
				[](Identifier* id) -> mold::DimRange
				{
					return mold::DimRange{
						id->get(),mold::Range{0,0}};
				});
				return out;
			}(args),
			slip::get_op(opcode)),
		slip::opnames.at(opcode)),
	opcode_(opcode),
	arg_ids_(
		[](std::vector<Identifier*> ids) -> std::vector<UID>
		{
			std::vector<UID> out(ids.size());
			std::transform(ids.begin(), ids.end(), out.begin(),
			[](Identifier * id) -> UID
			{
				return id->get_uid();
			});
			return out;
		}(args))
{
	// validate
	for (Identifier* arg : args)
	{
		UID uid = arg->get_uid();
		if (false == graph.has_node(uid))
		{
			throw MissingNodeError(graph.get_gid(), uid);
		}
	}
	graph_->add_func(opcode, this);
}

Functor::Functor (std::vector<IdRange> args,
	slip::OPCODE opcode, Graph& graph) :
	Identifier(&graph,
		new mold::Functor(
			[](std::vector<IdRange>& args) -> std::vector<mold::DimRange>
			{
				std::vector<mold::DimRange> out;
				std::transform(args.begin(), args.end(), std::back_inserter(out),
				[](IdRange id) -> mold::DimRange
				{
					return mold::DimRange{id.arg_->get(), id.drange_};
				});
				return out;
			}(args),
			slip::get_op(opcode)),
		slip::opnames.at(opcode)),
	opcode_(opcode),
	arg_ids_(
		[](std::vector<IdRange>& ids) -> std::vector<UID>
		{
			std::vector<UID> out(ids.size());
			std::transform(ids.begin(), ids.end(), out.begin(),
			[](IdRange& id) -> UID
			{
				return id.arg_->get_uid();
			});
			return out;
		}(args))
{
	// validate
	for (IdRange& arg : args)
	{
		UID uid = arg.arg_->get_uid();
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
