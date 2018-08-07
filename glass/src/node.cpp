#include <cstring>

#include "glass/node.hpp"

#include "soil/functor.hpp"

#ifdef GLASS_NODE_HPP

void save_info (tenncor::DataInfoPb& out, iNode* data)
{
	Shape shape = data->shape();
	std::vector<DimT> slist = shape.as_list();
	std::string encoding((char*) &slist[0], slist.size());
	// asserts DimT == uint8_t, ponder viability in future?
	encoding.insert(0, 1, shape.groups_encoding());
	out.set_shape(encoding);
	out.set_type(data->type());
}

DTYPE load_info (Shape& out, const tenncor::DataInfoPb& info)
{
	std::string shape = info.shape();
	uint8_t rank = shape.size();
	assert(rank <= rank_cap && false == shape.empty());
	uint8_t group = shape[0];
	char* ptr = &shape[1];
	std::vector<DimT> slist(ptr, ptr + (rank - 1));
	out = Shape(slist, group);
	return (DTYPE) info.type();
}

#define PACK_ARR(TYPE)\
TYPE* dptr = (TYPE*) raw;\
google::protobuf::RepeatedField<TYPE> field(\
	dptr, dptr + nelem);\
arr.mutable_data()->Swap(&field);\
out.mutable_data()->PackFrom(arr);

void save_node (tenncor::DataPb& out, iNode* data)
{
    Pool pool;
	std::shared_ptr<char> ptr = data->calculate(pool);
    char* raw = ptr.get();
	NElemT nelem = data->shape().n_elems();
	switch (data->type())
	{
		case INT8:
		case UINT8:
		{
			tenncor::ByteArr arr;
			arr.set_data(std::string(raw, nelem));
			out.mutable_data()->PackFrom(arr);
		}
		break;
		case DOUBLE:
		{
			tenncor::DoubleArr arr;
			PACK_ARR(double)
		}
		break;
		case FLOAT:
		{
			tenncor::FloatArr arr;
			PACK_ARR(float)
		}
		break;
		case INT16:
		{
			tenncor::Int32Arr arr;
			int16_t* dptr = (int16_t*) raw;
			std::vector<int16_t> temp(dptr, dptr + nelem);
			google::protobuf::RepeatedField<int32_t> field(
				temp.begin(), temp.end());
			arr.mutable_data()->Swap(&field);
			out.mutable_data()->PackFrom(arr);
		}
		break;
		case UINT16:
		{
			tenncor::Uint32Arr arr;
			uint16_t* dptr = (uint16_t*) raw;
			std::vector<uint16_t> temp(dptr, dptr + nelem);
			google::protobuf::RepeatedField<uint32_t> field(
				temp.begin(), temp.end());
			arr.mutable_data()->Swap(&field);
			out.mutable_data()->PackFrom(arr);
		}
		break;
		case INT32:
		{
			tenncor::Int32Arr arr;
			PACK_ARR(int32_t)
		}
		break;
		case UINT32:
		{
			tenncor::Uint32Arr arr;
			PACK_ARR(uint32_t)
		}
		break;
		case INT64:
		{
			tenncor::Int64Arr arr;
			PACK_ARR(int64_t)
		}
		break;
		case UINT64:
		{
			tenncor::Uint64Arr arr;
			PACK_ARR(uint64_t)
		}
		break;
		default:
			handle_error("serializing node with unsupported type");
	}
	save_info(*out.mutable_info(), data);
}

#define UNPACK_ARR(ARRTYPE)\
ARRTYPE arr;\
data.data().UnpackTo(&arr);\
auto field = arr.data();\
std::memcpy(dest, &field[0], nbytes);

DTYPE load_node (std::string& out, Shape& outshape, const tenncor::DataPb& data)
{
	DTYPE outtype = load_info(outshape, data.info());
	size_t nbytes = outshape.n_elems() * type_size(outtype);
	out = std::string(nbytes, 0);
	char* dest = &out[0];
	switch (outtype)
	{
		case INT8:
		case UINT8:
		{
			tenncor::ByteArr arr;
			data.data().UnpackTo(&arr);
			std::string src = arr.data();
			std::memcpy(dest, src.c_str(), nbytes);
		}
		break;
		case DOUBLE:
		{
			UNPACK_ARR(tenncor::DoubleArr)
		}
		break;
		case FLOAT:
		{
			UNPACK_ARR(tenncor::FloatArr)
		}
		break;
		case INT16:
		{
			tenncor::Int32Arr arr;
			data.data().UnpackTo(&arr);
			auto field = arr.data();
			std::vector<int16_t> temp(field.begin(), field.end());
			std::memcpy(dest, &temp[0], nbytes);
		}
		break;
		case UINT16:
		{
			tenncor::Uint32Arr arr;
			data.data().UnpackTo(&arr);
			auto field = arr.data();
			std::vector<uint16_t> temp(field.begin(), field.end());
			std::memcpy(dest, &temp[0], nbytes);
		}
		break;
		case INT32:
		{
			UNPACK_ARR(tenncor::Int32Arr)
		}
		break;
		case UINT32:
		{
			UNPACK_ARR(tenncor::Uint32Arr)
		}
		break;
		case INT64:
		{
			UNPACK_ARR(tenncor::Int64Arr)
		}
		break;
		case UINT64:
		{
			UNPACK_ARR(tenncor::Uint64Arr)
		}
		break;
		default:
			handle_error("deserializing node with unsupported type");
	}
	return outtype;
}

std::list<iNode*> order_nodes (const Session& in)
{
	std::list<iNode*> q;
	for (auto npair : in.nodes_)
	{
		auto ref = npair.second;
		if (false == ref.expired())
		{
			q.push_back(ref.lock().get());
		}
	}
	std::list<iNode*> nodes;
	std::unordered_set<iNode*> visited;
	while (false == q.empty())
	{
		iNode* node = q.front();
		q.pop_front();
		if (visited.end() == visited.find(node))
		{
			if (Functor* f = dynamic_cast<Functor*>(node))
			{
				auto args = f->get_refs();
				q.insert(q.end(), args.begin(), args.end());
			}
			nodes.push_back(node);
			visited.emplace(node);
		}
	}
	return nodes;
}

std::list<iNode*> order_nodes (const Session& in,
	std::unordered_map<iNode*,uint32_t>& nodemap)
{
	std::list<iNode*> q;
	for (auto npair : in.nodes_)
	{
		auto ref = npair.second;
		if (false == ref.expired())
		{
			q.push_back(ref.lock().get());
		}
	}
	std::list<iNode*> nodes;
	while (false == q.empty())
	{
		iNode* node = q.front();
		q.pop_front();
		if (nodemap.end() == nodemap.find(node))
		{
			if (Functor* f = dynamic_cast<Functor*>(node))
			{
				auto args = f->get_refs();
				q.insert(q.end(), args.begin(), args.end());
			}
			nodemap[node] = nodes.size();
			nodes.push_back(node);
		}
	}
	return nodes;
}

#endif
