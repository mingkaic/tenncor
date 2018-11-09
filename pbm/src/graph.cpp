#include <unordered_set>
#include <list>
#include <queue>
#include <chrono>

#include "err/log.hpp"

#include "adhoc/llo/api.hpp"

#include "pbm/graph.hpp"
#include "pbm/source.hpp"

#ifdef PBM_GRAPH_HPP

static std::string make_uid (void* ptr, llo::EngineT& engine)
{
	static std::uniform_int_distribution<short> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) ptr;

	for (size_t i = 0; i < 16; i++)
	{
		short token = tok_dist(engine);
		ss << std::hex << token;
	}
	return ss.str();
}

void save_coord (google::protobuf::RepeatedField<double>* coord,
	const ade::CoordPtrT& mapper)
{
	mapper->access([coord](const ade::MatrixT& mat)
	{
		for (uint8_t i = 0; i < ade::mat_dim; ++i)
		{
			for (uint8_t j = 0; j < ade::mat_dim; ++j)
			{
				(*coord->Add()) = mat[i][j];
			}
		}
	});
}

ade::CoordPtrT load_coord (
	const google::protobuf::RepeatedField<double>& coord)
{
	if (ade::mat_dim * ade::mat_dim != coord.size())
	{
		err::fatal("cannot deserialize non-matrix coordinate map");
	}
	return std::make_shared<ade::CoordMap>(
		[&](ade::MatrixT fwd)
		{
			for (uint8_t i = 0; i < ade::mat_dim; ++i)
			{
				for (uint8_t j = 0; j < ade::mat_dim; ++j)
				{
					fwd[i][j] = coord[i * ade::mat_dim + j];
				}
			}
		});
}

struct GraphDFSOrder final : public ade::iTraveler
{
	/// Implemenation of iTraveler
	void visit (ade::Tensor* leaf) override
	{
		if (visited_.end() == visited_.find(leaf))
		{
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	/// Implemenation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func))
		{
			funcs_.push_back(func);
			visited_.emplace(func);

			ade::ArgsT children = func->get_children();
			for (auto& child : children)
			{
				child.tensor_->accept(*this);
			}
		}
	}

	// List of leaves visited (left to right)
	std::list<ade::Tensor*> leaves_;

	// List of functions visited (by depth-first)
	std::list<ade::iFunctor*> funcs_;

	// Visited nodes
	std::unordered_set<ade::iTensor*> visited_;
};

void save_graph (tenncor::Graph& out, std::vector<llo::DataNode>& roots)
{
	ade::GraphStat stat;
	GraphDFSOrder order;

	std::vector<const llo::EvalCtx*> contexas(roots.size());
	std::transform(roots.begin(), roots.end(), contexas.begin(),
	[&](llo::DataNode& tptr)
	{
		tptr.tensor_->accept(stat);
		tptr.tensor_->accept(order);
		return &tptr.ctx_;
	});
	llo::EvalCtx global_ctx(contexas);

	// sort functions from the root with the smallest subtree to the largest
	// this ensures every children of a node appears before the parent,
	// as is the order of node creations
	order.funcs_.sort(
		[&](ade::iTensor* a, ade::iTensor* b)
		{
			return stat.graphsize_[a] < stat.graphsize_[b];
		});

	std::vector<ade::iFunctor*> funcs(
		order.funcs_.begin(), order.funcs_.end());
	std::vector<ade::Tensor*> leaves;
	std::copy_if(order.leaves_.begin(), order.leaves_.end(),
		std::back_inserter(leaves),
		[&](ade::Tensor* leaf)
		{
			return global_ctx.srcs_.end() != global_ctx.srcs_.find(leaf);
		});

	// all nodes in leaf appear before funcs
	std::unordered_map<ade::iTensor*,size_t> ordermap;
	size_t nleaves = leaves.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		llo::iSource* source = global_ctx.srcs_[leaves[i]].get();
		ade::Tensor* tens = source->inner().get();
		ordermap[tens] = i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Source* src = pb_node->mutable_source();
		save_data(src, source);
	}
	for (size_t i = 0, n = funcs.size(); i < n; ++i)
	{
		ade::iFunctor* f = funcs[i];
		ordermap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Functor* func = pb_node->mutable_functor();
		func->set_opname(f->get_opcode().name_);
		const ade::ArgsT& children = f->get_children();
		for (auto& child : children)
		{
			tenncor::NodeArg* arg = func->add_args();
			ade::iTensor* tens = child.tensor_.get();
			if (tens == ade::Tensor::SYMBOLIC_ONE.get())
			{
				arg->set_idx(-2);
			}
			else if (tens == ade::Tensor::SYMBOLIC_ZERO.get())
			{
				arg->set_idx(-1);
			}
			else
			{
				arg->set_idx(ordermap[tens]);
			}
			save_coord(arg->mutable_coord(), child.mapper_);
		}
	}
	out.set_id(make_uid(&out, llo::get_engine()));
}

std::vector<llo::DataNode> load_graph (const tenncor::Graph& in)
{
	auto nodes = in.nodes();
	std::vector<llo::DataNode> outvec;
	for (const tenncor::Node& node : nodes)
	{
		if (node.has_source())
		{
			const tenncor::Source& source = node.source();
			outvec.push_back(load_source(source));
		}
		else
		{
			tenncor::Functor func = node.functor();
			auto nodeargs = func.args();
			ade::ArgsT args;
			std::vector<const llo::EvalCtx*> contexas;
			for (auto nodearg : nodeargs)
			{
				int32_t i = nodearg.idx();
				ade::CoordPtrT coord = load_coord(nodearg.coord());
				if (i >= 0)
				{
					args.push_back({coord, outvec[i].tensor_});
					contexas.push_back(&outvec[i].ctx_);
				}
				else if (-2 == i)
				{
					args.push_back({coord, ade::Tensor::SYMBOLIC_ONE});
				}
				else if (-1 == i)
				{
					args.push_back({coord, ade::Tensor::SYMBOLIC_ZERO});
				}
				else
				{
					err::fatalf("cannot find tensor of index %d", i);
				}
			}
			outvec.push_back(llo::DataNode{llo::EvalCtx(contexas),
				ade::Functor::get(age::make_code(
					age::name_op(func.opname())), args)});
		}
	}
	return outvec;
}

#endif
