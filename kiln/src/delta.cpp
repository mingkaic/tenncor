//
//  delta.cpp
//  kiln
//

#include <algorithm>
#include <stack>
#include <queue>

#include "kiln/delta.hpp"
#include "kiln/graph.hpp"
#include "kiln/constant.hpp"
#include "kiln/functor.hpp"
#include "kiln/operators.hpp"

#ifdef KILN_DELTA_HPP

namespace kiln
{

using IdSetT = std::unordered_set<UID>;

using UIDGreater = std::greater<UID>;

using BotUpPathT = std::priority_queue<UID,std::vector<UID>,UIDGreater>;

static void get_paths (std::unordered_set<UID>& reachable,
	Identifier* root, UID target, Graph* graph)
{
	assert(nullptr != graph);
	if (target > root->get_uid())
	{
		return;
	}
	std::unordered_set<UID> visited;

	std::stack<Identifier*> stk;
	std::list<UID> path = {0};
	stk.push(root);

	while (false == stk.empty())
	{
		Identifier* top = stk.top();
		stk.pop();
		if (nullptr == top)
		{
			path.pop_back();
		}
		else
		{
			UID id = top->get_uid();
			path.back() = id;
			visited.emplace(id);
			if (id == target)
			{
				// add path
				for (auto it = path.begin(), et = --path.end();
					it != et; ++it)
				{
					reachable.emplace(*it);
				}
			}
			else if (id > target)
			{
				auto args = top->get_args();
				stk.push(nullptr);
				path.push_back(0);
				for (UIDRange& arg : args)
				{
					if (visited.end() == visited.find(arg.uid_))
					{
						stk.push(graph->get_node(arg.uid_));
					}
					else if (reachable.end() != reachable.find(arg.uid_) ||
						arg.uid_ == target)
					{
						// add path
						for (auto it = path.begin(), et = --path.end();
							it != et; ++it)
						{
							reachable.emplace(*it);
						}
					}
				}
			}
		}
	}
}

Identifier* delta (Identifier* root, Identifier* wrt)
{
	// sanity check
	if (root == nullptr || wrt == nullptr)
	{
		return nullptr;
	}
	if (false == root->has_data() ||
		false == wrt->has_data())
	{
		throw mold::UninitializedError();
	}
	if (nullptr != dynamic_cast<Constant*>(wrt))
	{
		throw std::logic_error("deriving with respect to a constant");
	}

	// optimizations
	if (root == wrt)
	{
		return make_one(wrt);
	}
	if (nullptr == dynamic_cast<Functor*>(root) || root->graph_ != wrt->graph_)
	{
		return make_zero(wrt);
	}

	UID target = wrt->get_uid();
	Graph* graph = root->graph_;
	if (nullptr == graph)
	{
		throw std::exception(); // todo: add context
	}
	std::unordered_set<UID> reachable;
	get_paths(reachable, root, target, graph);
	if (reachable.empty())
	{
		return make_zero(wrt);
	}
	// asserts lower UID always appear lower in the graph
	BotUpPathT path;
	for (UID id : reachable)
	{
		path.push(id);
	}
	Constant* one = make_one(wrt);
	std::unordered_map<UID,Identifier*> backs = {{target, one}};
	while (false == path.empty())
	{
		UID id = path.top();
		path.pop();
		Functor* f = static_cast<Functor*>(graph->get_node(id));
		std::vector<UIDRange> ids = f->get_args();
		GradArgsT args;
		std::transform(ids.begin(), ids.end(), std::back_inserter(args),
		[&](UIDRange& id) -> std::pair<IdRange,IdRange>
		{
			Identifier* fwd = graph->get_node(id.uid_);
			Identifier* bwd = nullptr;
			auto it = backs.find(id.uid_);
			if (backs.end() != it)
			{
				bwd = it->second;
			}
			return {IdRange{fwd, id.range_},
				IdRange{bwd, id.range_}};
		});
		backs[id] = grad_op.at(f->opcode_)(one, args);
	}
	Identifier* out = backs[root->get_uid()];
	if (nullptr == out)
	{
		out = make_zero(wrt);
	}
	else
	{
		clay::Shape shape = out->get()->get_shape();
		clay::Shape base = wrt->get()->get_shape();
		if (false == shape.is_compatible_with(base) &&
			shape.rank() > 1 && shape.at(1) == base.n_elems())
		// this is an unreliable way of detecting jacobian node
		// todo: improve this check
		{
			out = reduce_sum(out, (uint64_t) 0);
			std::vector<size_t> blist = base.as_list();
			out = reshape(out, std::vector<uint64_t>(
				blist.begin(), blist.end()));
		}
	}
	return out;
}

}

#endif
