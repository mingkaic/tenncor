#include <unordered_set>
#include <list>
#include <queue>
#include <chrono>

#include "ade/log.hpp"

#include "llo/api.hpp"

#include "pbm/graph.hpp"

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

#define PACK_DATA(TYPE)\
TYPE* ptr = (TYPE*) data.data_.get();\
google::protobuf::RepeatedField<TYPE> vec(ptr, ptr + nelems);\
arr->mutable_data()->Swap(&vec);

static void save_data (tenncor::Source* out, llo::GenericData& data)
{
	size_t nelems = data.shape_.n_elems();
	switch (data.dtype_)
	{
		case llo::DOUBLE:
		{
			auto arr = out->mutable_double_arrs();
			PACK_DATA(double)
		}
		break;
		case llo::FLOAT:
		{
			auto arr = out->mutable_float_arrs();
			PACK_DATA(float)
		}
		break;
		case llo::INT8:
		{
			auto arr = out->mutable_sbyte_arrs();
			char* ptr = data.data_.get();
			arr->set_data(std::string(ptr, ptr + nelems));
		}
		break;
		case llo::UINT8:
		{
			auto arr = out->mutable_ubyte_arrs();
			char* ptr = data.data_.get();
			arr->set_data(std::string(ptr, ptr + nelems));
		}
		break;
		case llo::INT16:
		{
			auto arr = out->mutable_sshort_arrs();
			int16_t* ptr = (int16_t*) data.data_.get();
			std::vector<int16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<int32_t> vec(
				temp.begin(), temp.end());
			arr->mutable_data()->Swap(&vec);
		}
		break;
		case llo::INT32:
		{
			auto arr = out->mutable_sint_arrs();
			PACK_DATA(int32_t)
		}
		break;
		case llo::INT64:
		{
			auto arr = out->mutable_slong_arrs();
			PACK_DATA(int64_t)
		}
		break;
		case llo::UINT16:
		{
			auto arr = out->mutable_ushort_arrs();
			uint16_t* ptr = (uint16_t*) data.data_.get();
			std::vector<uint16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<uint32_t> vec(
				temp.begin(), temp.end());
			arr->mutable_data()->Swap(&vec);
		}
		break;
		case llo::UINT32:
		{
			auto arr = out->mutable_uint_arrs();
			PACK_DATA(uint32_t)
		}
		break;
		case llo::UINT64:
		{
			auto arr = out->mutable_ulong_arrs();
			PACK_DATA(uint64_t)
		}
		break;
		default:
			ade::error("cannot serialize badly typed node... skipping");
	}
}

#undef PACK_DATA

static void save_meta (google::protobuf::RepeatedField<uint32_t>* meta,
	llo::iEvaluable* eval, ade::OPCODE op)
{
	switch (op)
	{
		case ade::FLIP:
		case ade::N_DIMS:
		{
			auto ev = static_cast<llo::FuncWrapper<uint8_t>*>(eval);
			*(meta->Add()) = std::get<0>(ev->meta());
		}
		break;
		case ade::MATMUL:
		{
			auto ev = static_cast<llo::FuncWrapper<uint8_t,uint8_t>*>(eval);
			*(meta->Add()) = std::get<0>(ev->meta());
			*(meta->Add()) = std::get<1>(ev->meta());
		}
		break;
		default: break; // no meta
	}
}

static void save_meta (google::protobuf::RepeatedField<uint32_t>* meta,
	ade::iFunctor* f, ade::OPCODE op)
{
	switch (op)
	{
		case ade::MATMUL:
		{
			auto ev = static_cast<ade::Functor<
				ade::MATMUL,uint8_t,uint8_t>*>(f);
			*(meta->Add()) = std::get<0>(ev->meta());
			*(meta->Add()) = std::get<1>(ev->meta());
		}
		break;
		case ade::PERMUTE:
		{
			auto ev = static_cast<ade::Functor<
				ade::PERMUTE,std::vector<uint8_t>>*>(f);
			std::vector<uint8_t> slist = std::get<0>(ev->meta());
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		case ade::EXTEND:
		{
			auto ev = static_cast<ade::Functor<
				ade::EXTEND,std::vector<ade::DimT>>*>(f);
			std::vector<ade::DimT> slist = std::get<0>(ev->meta());
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		case ade::RESHAPE:
		{
			auto ev = static_cast<ade::Functor<
				ade::RESHAPE,std::vector<ade::DimT>>*>(f);
			std::vector<ade::DimT> slist = std::get<0>(ev->meta());
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		default: break; // no meta
	}
}

struct GraphStat final : public ade::Traveler
{
	void visit (ade::Tensor* leaf) override
	{
		if (visited_.end() == visited_.find(leaf))
		{
			leaves_.push_back(leaf);
			visited_.emplace(leaf);
		}
	}

	void visit (ade::iFunctor* func) override
	{
		if (graphsize_.end() == graphsize_.find(func))
		{
			order_.push_back(func);
			auto children = func->get_children();
			size_t ngraph = 0;
			for (ade::iTensor* child : children)
			{
				if (graphsize_.end() == graphsize_.find(child))
				{
					child->accept(*this);
				}
				auto childinfo = graphsize_.find(child);
				if (graphsize_.end() != childinfo &&
					childinfo->second > ngraph)
				{
					ngraph = childinfo->second;
				} // else child is leaf
			}
			graphsize_[func] = ngraph + 1;
		}
	}

	std::vector<ade::Tensor*> leaves_;

	// ensure we don't serialize leaves twice
	std::unordered_set<ade::Tensor*> visited_;

	// store list of funcs to ensure determinisitc ordering
	std::list<ade::iFunctor*> order_;

	std::unordered_map<ade::iTensor*,size_t> graphsize_;
};

void save_graph (tenncor::Graph& out, std::vector<llo::DataNode>& roots)
{
	std::vector<const llo::EvalCtx*> contexas(roots.size());
	std::transform(roots.begin(), roots.end(), contexas.begin(),
	[](llo::DataNode& tptr)
	{
		return &tptr.ctx_;
	});
	llo::EvalCtx global_ctx(contexas);

	GraphStat stat;
	for (llo::DataNode& tptr : roots)
	{
		tptr.tensor_->accept(stat);
	}

	std::list<ade::iTensor*> funcs(stat.order_.begin(), stat.order_.end());
	// sort functions from the root with the smallest subgraph to the largest
	// this ensures every children of a node appears before the parent,
	// as is the order of node creations
	funcs.sort(
	[&stat](ade::iTensor* a, ade::iTensor* b)
	{
		return stat.graphsize_[a] < stat.graphsize_[b];
	});

	// all nodes in leaf appear before funcs
	std::unordered_map<ade::iTensor*,size_t> ordermap;
	size_t nleaves = stat.leaves_.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		ade::Tensor* leaf = stat.leaves_[i];
		ordermap[leaf] = i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Source* src = pb_node->mutable_source();
		auto vec = leaf->shape().as_list();
		std::string shape(vec.begin(), vec.end());
		src->set_shape(shape);
		auto srcinfo = global_ctx.srcs_.find(leaf);
		if (global_ctx.srcs_.end() != srcinfo)
		{
			llo::GenericData data = srcinfo->second->data(
				srcinfo->second->native_type());
			save_data(src, data);
		}
	}
	auto it = funcs.begin();
	for (size_t i = 0, n = funcs.size(); i < n; ++i)
	{
		ade::iFunctor* f = static_cast<ade::iFunctor*>(*(it++));
		ordermap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Functor* func = pb_node->mutable_functor();
		ade::OPCODE op = f->get_code();
		func->set_opcode(op);
		std::vector<ade::iTensor*> children = f->get_children();
		google::protobuf::RepeatedField<uint32_t> indices;
		for (ade::iTensor* child : children)
		{
			auto it = ordermap.find(child);
			assert(ordermap.end() != it);
			indices.Add(it->second);
		}
		func->mutable_args()->Swap(&indices);
		auto it = global_ctx.funks_.find(f);
		if (global_ctx.funks_.end() != it)
		{
			save_meta(func->mutable_meta(), it->second.get(), op);
		}
		else
		{
			save_meta(func->mutable_meta(), f, op);
		}
	}
	out.set_id(make_uid(&out, llo::get_engine()));
}

static ade::Shape load_shape (std::string sstr)
{
	std::vector<ade::DimT> slist(sstr.begin(), sstr.end());
	return ade::Shape(slist);
}

#define UNPACK_SOURCE(TYPE)\
auto vec = arr.data();\
return llo::Source<TYPE>::get(shape,\
	std::vector<TYPE>(vec.begin(), vec.end()));

static llo::DataNode load_source (const tenncor::Source& source)
{
	ade::Shape shape = load_shape(source.shape());
	switch (source.data_case())
	{
		case tenncor::Source::DataCase::kDoubleArrs:
		{
			auto arr = source.double_arrs();
			UNPACK_SOURCE(double)
		}
		case tenncor::Source::DataCase::kFloatArrs:
		{
			auto arr = source.float_arrs();
			UNPACK_SOURCE(float)
		}
		case tenncor::Source::DataCase::kSbyteArrs:
		{
			auto arr = source.sbyte_arrs();
			UNPACK_SOURCE(int8_t)
		}
		case tenncor::Source::DataCase::kUbyteArrs:
		{
			auto arr = source.ubyte_arrs();
			UNPACK_SOURCE(uint8_t)
		}
		break;
		case tenncor::Source::DataCase::kSshortArrs:
		{
			auto arr = source.sshort_arrs();
			UNPACK_SOURCE(int16_t)
		}
		break;
		case tenncor::Source::DataCase::kSintArrs:
		{
			auto arr = source.sint_arrs();
			UNPACK_SOURCE(int32_t)
		}
		break;
		case tenncor::Source::DataCase::kSlongArrs:
		{
			auto arr = source.slong_arrs();
			UNPACK_SOURCE(int64_t)
		}
		break;
		case tenncor::Source::DataCase::kUshortArrs:
		{
			auto arr = source.ushort_arrs();
			UNPACK_SOURCE(uint16_t)
		}
		break;
		case tenncor::Source::DataCase::kUintArrs:
		{
			auto arr = source.uint_arrs();
			UNPACK_SOURCE(uint32_t)
		}
		break;
		case tenncor::Source::DataCase::kUlongArrs:
		{
			auto arr = source.ulong_arrs();
			UNPACK_SOURCE(uint64_t)
		}
		break;
		default:
			ade::fatalf("cannot load source"); // todo: make more informative
	}
}

#undef UNPACK_SOURCE

static llo::DataNode load_op (ade::OPCODE opcode,
	std::vector<llo::DataNode>& args,
	google::protobuf::RepeatedField<uint32_t> meta)
{
	switch (opcode)
	{
		case ade::FLIP:
			return llo::flip(args[0], meta[0]);
		case ade::N_DIMS:
			return llo::n_dims(args[0], meta[0]);
		case ade::MATMUL:
			return llo::matmul(args[0], args[1],
				meta[0], meta[1]);
		case ade::PERMUTE:
			return llo::permute(args[0],
				std::vector<uint8_t>(meta.begin(), meta.end()));
		case ade::EXTEND:
			return llo::extend(args[0],
				std::vector<uint8_t>(meta.begin(), meta.end()));
		case ade::RESHAPE:
			return llo::reshape(args[0],
				std::vector<uint8_t>(meta.begin(), meta.end()));
		default: break;
	}
	std::vector<ade::Tensorptr> tens;
	std::vector<const llo::EvalCtx*> contexas;
	for (llo::DataNode& node : args)
	{
		tens.push_back(node.tensor_);
		contexas.push_back(&node.ctx_);
	}
	return llo::DataNode{llo::EvalCtx(contexas),
		ade::runtime_functor(opcode, tens)};
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
			auto argidx = func.args();
			std::vector<llo::DataNode> args;
			for (uint32_t i : argidx)
			{
				args.push_back(outvec[i]);
			}
			outvec.push_back(load_op((ade::OPCODE)
				func.opcode(), args, func.meta()));
		}
	}
	return outvec;
}
