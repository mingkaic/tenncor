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

UIDRange IdRange::get_uid (void) const
{
	return UIDRange{arg_->get_uid(), drange_};
}

Functor::Functor (std::vector<Identifier*> args,
	slip::OPCODE opcode, Graph& graph) :
	Identifier(&graph,
		new mold::Functor(
			[](std::vector<Identifier*> args) -> std::vector<mold::NodeRange>
			{
				std::vector<mold::NodeRange> out;
				std::transform(args.begin(), args.end(), std::back_inserter(out),
				[](Identifier* id) -> mold::NodeRange
				{
					return mold::NodeRange{
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
			[](std::vector<IdRange>& args) -> std::vector<mold::NodeRange>
			{
				std::vector<mold::NodeRange> out;
				std::transform(args.begin(), args.end(), std::back_inserter(out),
				[](IdRange id) -> mold::NodeRange
				{
					return mold::NodeRange{id.arg_->get(), id.drange_};
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

std::vector<UIDRange> Functor::get_args (void) const
{
	size_t n = arg_ids_.size();
	std::vector<UIDRange> out;
	std::vector<mold::Range> ranges = static_cast<mold::Functor*>(get())->get_ranges();
	assert(n == ranges.size());
	for (size_t i = 0; i < n; ++i)
	{
		out.push_back({arg_ids_[i], ranges[i]});
	}
	return out;
}

}

#endif
