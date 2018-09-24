#include <unordered_set>
#include <list>
#include <queue>

#include "util/rand.hpp"
#include "util/error.hpp"

#include "llo/api.hpp"

#include "pbm/graph.hpp"

#define PACK_DATA(TYPE)\
TYPE* ptr = (TYPE*) data.data_.get();\
google::protobuf::RepeatedField<TYPE> vec(ptr, ptr + nelems);\
arr.mutable_data()->Swap(&vec);\
out->PackFrom(arr);

static void save_data (google::protobuf::Any* out, llo::GenericData& data)
{
	size_t nelems = data.shape_.n_elems();
	switch (data.dtype_)
	{
		case llo::DOUBLE:
		{
			tenncor::DoubleArr arr;
			PACK_DATA(double)
		}
		break;
		case llo::FLOAT:
		{
			tenncor::FloatArr arr;
			PACK_DATA(float)
		}
		break;
		case llo::INT8:
		case llo::UINT8:
		{
			tenncor::ByteArr arr;
			char* ptr = data.data_.get();
			arr.set_data(std::string(ptr, ptr + nelems));
			out->PackFrom(arr);
		}
		break;
		case llo::INT16:
		{
			tenncor::Int32Arr arr;
			int16_t* ptr = (int16_t*) data.data_.get();
			std::vector<int16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<int32_t> vec(
				temp.begin(), temp.end());
			arr.mutable_data()->Swap(&vec);
			out->PackFrom(arr);
		}
		break;
		case llo::UINT16:
		{
			tenncor::Uint32Arr arr;
			uint16_t* ptr = (uint16_t*) data.data_.get();
			std::vector<uint16_t> temp(ptr, ptr + nelems);
			google::protobuf::RepeatedField<uint32_t> vec(
				temp.begin(), temp.end());
			arr.mutable_data()->Swap(&vec);
			out->PackFrom(arr);
		}
		break;
		case llo::INT32:
		{
			tenncor::Int32Arr arr;
			PACK_DATA(int32_t)
		}
		break;
		case llo::INT64:
		{
			tenncor::Int64Arr arr;
			PACK_DATA(int64_t)
		}
		break;
		case llo::UINT32:
		{
			tenncor::Uint32Arr arr;
			PACK_DATA(uint32_t)
		}
		break;
		case llo::UINT64:
		{
			tenncor::Uint64Arr arr;
			PACK_DATA(uint64_t)
		}
		break;
		default:
			util::handle_error("failed to serialize badly typed node"); // todo: make warn instead
	}
}

#undef PACK_DATA

static void save_meta (google::protobuf::RepeatedField<uint32_t>* meta,
	ade::iFunctor* f, ade::OPCODE op)
{
	switch (op)
	{
		case ade::FLIP:
		case ade::N_DIMS:
		{
			auto ev = static_cast<llo::DirectWrapper<uint8_t>*>(f);
			*(meta->Add()) = std::get<0>(ev->args_);
		}
		break;
		case ade::MATMUL:
		{
			auto ev = static_cast<llo::DirectWrapper<uint8_t,uint8_t>*>(f);
			*(meta->Add()) = std::get<0>(ev->args_);
			*(meta->Add()) = std::get<1>(ev->args_);
		}
		break;
		case ade::PERMUTE:
		{
			auto ev = static_cast<llo::DirectWrapper<std::vector<uint8_t>>*>(f);
			std::vector<uint8_t> slist = std::get<0>(ev->args_);
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		case ade::EXTEND:
		{
			auto ev = static_cast<
				ade::Functor<ade::EXTEND,std::vector<ade::DimT>>*>(f);
			std::vector<ade::DimT> slist = std::get<0>(ev->meta_);
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		case ade::RESHAPE:
		{
			auto ev = static_cast<
				ade::Functor<ade::RESHAPE,std::vector<ade::DimT>>*>(f);
			std::vector<ade::DimT> slist = std::get<0>(ev->meta_);
			google::protobuf::RepeatedField<uint32_t> vec(
				slist.begin(), slist.end());
			meta->Swap(&vec);
		}
		break;
		default: break; // no meta
	}
}

void save_graph (tenncor::Graph& out, std::vector<ade::Tensorptr>& roots)
{
	ade::iTensor* iter;
	std::unordered_set<ade::iTensor*> visited;
	std::vector<ade::iTensor*> leaves;
	std::vector<ade::iFunctor*> order;
	for (ade::Tensorptr& tptr : roots)
	{
		std::list<ade::iFunctor*> appearance;
		std::queue<ade::iTensor*> q;
		q.push(tptr.get());
		while (false == q.empty())
		{
			iter = q.front();
			q.pop();
			if (visited.end() == visited.find(iter))
			{
				if (ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(iter))
				{
					auto children = f->get_refs();
					for (auto child : children)
					{
						q.push(child);
					}
					appearance.push_back(f);
				}
				else
				{
					leaves.push_back(iter);
				}
				visited.emplace(iter);
			}
		}
		order.insert(order.end(), appearance.rbegin(), appearance.rend());
	}
	// order guarantees for any index i, all children of node i is in order[:i]
	// all nodes in leaf are some children of nodes in order
	std::unordered_map<ade::iTensor*,size_t> imap;
	size_t nleaves = leaves.size();
	for (size_t i = 0; i < nleaves; ++i)
	{
		ade::iTensor* node = leaves[i];
		imap[node] = i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Source* src = pb_node->mutable_source();
		if (llo::iSource* eval = dynamic_cast<llo::iSource*>(node))
		{
			llo::GenericData data = eval->evaluate(eval->native_type());
			save_data(src->mutable_data(), data);
			src->set_type(data.dtype_);
		}
		else
		{
			src->set_type(llo::BAD);
		}
		auto vec = node->shape().as_list();
		std::string shape(vec.begin(), vec.end());
		src->set_shape(shape);
	}
	for (size_t i = 0, n = order.size(); i < n; ++i)
	{
		ade::iFunctor* f = order[i];
		imap[f] = nleaves + i;

		tenncor::Node* pb_node = out.add_nodes();
		tenncor::Functor* func = pb_node->mutable_functor();
		ade::OPCODE op = f->get_code();
		func->set_opcode(op);
		std::vector<ade::iTensor*> children = f->get_refs();
		google::protobuf::RepeatedField<uint32_t> indices;
		for (ade::iTensor* child : children)
		{
			auto it = imap.find(child);
			assert(imap.end() != it);
			indices.Add(it->second);
		}
		func->mutable_args()->Swap(&indices);
		save_meta(func->mutable_meta(), f, op);
	}
	out.set_id(util::make_uid(&out));
}

static ade::Shape load_shape (std::string sstr)
{
	std::vector<ade::DimT> slist(sstr.begin(), sstr.end());
	return ade::Shape(slist);
}

#define UNPACK_SOURCE(TYPE)\
source.data().UnpackTo(&arr);\
auto vec = arr.data();\
return llo::Source<TYPE>::get(shape,\
	std::vector<TYPE>(vec.begin(), vec.end()));

static ade::Tensorptr load_source (tenncor::Source& source)
{
	ade::Shape shape = load_shape(source.shape());
	switch (source.type())
	{
		case llo::DOUBLE:
		{
			tenncor::DoubleArr arr;
			UNPACK_SOURCE(double)
		}
		case llo::FLOAT:
		{
			tenncor::FloatArr arr;
			UNPACK_SOURCE(float)
		}
		case llo::INT8:
		{
			tenncor::ByteArr arr;
			UNPACK_SOURCE(int8_t)
		}
		case llo::UINT8:
		{
			tenncor::ByteArr arr;
			UNPACK_SOURCE(uint8_t)
		}
		break;
		case llo::INT16:
		{
			tenncor::Int32Arr arr;
			UNPACK_SOURCE(int16_t)
		}
		break;
		case llo::UINT16:
		{
			tenncor::Uint32Arr arr;
			UNPACK_SOURCE(uint16_t)
		}
		break;
		case llo::INT32:
		{
			tenncor::Int32Arr arr;
			UNPACK_SOURCE(int32_t)
		}
		break;
		case llo::INT64:
		{
			tenncor::Int64Arr arr;
			UNPACK_SOURCE(int64_t)
		}
		break;
		case llo::UINT32:
		{
			tenncor::Uint32Arr arr;
			UNPACK_SOURCE(uint32_t)
		}
		break;
		case llo::UINT64:
		{
			tenncor::Uint64Arr arr;
			UNPACK_SOURCE(uint64_t)
		}
		break;
		default:
			util::handle_error("failed to deserialize badly typed node"); // todo: make warn instead
	}
	return nullptr;
}

#undef UNPACK_SOURCE

static ade::Tensorptr load_op (ade::OPCODE opcode,
	std::vector<ade::Tensorptr>& args,
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
	return ade::runtime_functor(opcode, args);
}

std::vector<ade::Tensorptr> load_graph (const tenncor::Graph& in)
{
	auto nodes = in.nodes();
	std::vector<ade::Tensorptr> outvec;
	for (const tenncor::Node& node : nodes)
	{
		if (node.has_source())
		{
			tenncor::Source source = node.source();
			if (source.type() != llo::BAD)
			{
				outvec.push_back(load_source(source));
			}
			else
			{
				outvec.push_back(ade::Tensor::get(
					load_shape(source.shape())));
			}
		}
		else
		{
			tenncor::Functor func = node.functor();
			auto argidx = func.args();
			std::vector<ade::Tensorptr> args;
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
