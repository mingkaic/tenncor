#include <unordered_set>
#include <list>
#include <queue>
#include <chrono>

#include "ade/log/log.hpp"

#include "llo/api.hpp"

#include "pbm/graph.hpp"
#include "pbm/source.hpp"

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

ade::CoordPtrT load_coord (const google::protobuf::RepeatedField<double>& coord)
{
	if (ade::mat_dim * ade::mat_dim != coord.size())
	{
		ade::fatal("cannot deserialize non-matrix coordinate map");
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

void save_graph (tenncor::Graph& out, std::vector<llo::DataNode>& roots)
{
	llo::GraphStat stat(roots);

	// all nodes in leaf appear before funcs
	std::unordered_map<ade::iTensor*,size_t> ordermap;
	size_t nleaves = stat.leaves_.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		llo::iSource* source = stat.leaves_[i];
		ade::Tensor* tens = source->inner().get();
		ordermap[tens] = i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Source* src = pb_node->mutable_source();
		save_data(src, source);
	}
	auto it = stat.funcs_.begin();
	for (size_t i = 0, n = stat.funcs_.size(); i < n; ++i)
	{
		ade::iFunctor* f = *(it++);
		ordermap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Functor* func = pb_node->mutable_functor();
		func->set_opname(ade::opname(f->get_code()));
		const ade::ArgsT& children = f->get_children();
		for (auto& child : children)
		{
			tenncor::NodeArg* arg = func->add_args();
			ade::iTensor* tens = child.second.get();
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
			save_coord(arg->mutable_coord(), child.first);
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
					ade::fatalf("cannot find tensor of index %d", i);
				}
			}
			outvec.push_back(llo::DataNode{llo::EvalCtx(contexas),
				ade::Functor::get(ade::name_op(func.opname()), args)});
		}
	}
	return outvec;
}
