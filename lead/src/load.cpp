//
//  load.cpp
//  lead
//

#include "lead/load.hpp"
#include "lead/include/packer.hpp"
#include "lead/include/pb_build.hpp"

#include "wire/functor.hpp"
#include "wire/constant.hpp"
#include "wire/variable.hpp"
#include "wire/placeholder.hpp"

#ifdef LEAD_LOAD_HPP

namespace lead
{

static inline void placeholder_assign (wire::Placeholder* place,
    const tenncor::TensorPb& tpb)
{
	switch (tpb.type())
	{
		case tenncor::DOUBLE:
		{
			tenncor::DoubleArr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<double>& vec = arr.data();
			*place = std::vector<double>(vec.begin(), vec.end());
		}
		break;
		case tenncor::FLOAT:
		{
			tenncor::FloatArr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<float>& vec = arr.data();
			*place = std::vector<float>(vec.begin(), vec.end());
		}
		break;
		case tenncor::INT8:
		{
			tenncor::ByteArr arr;
			tpb.data().UnpackTo(&arr);
			std::string vec = arr.data();
			*place = std::vector<int8_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::UINT8:
		{
			tenncor::ByteArr arr;
			tpb.data().UnpackTo(&arr);
			std::string vec = arr.data();
			*place = std::vector<uint8_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::INT16:
		{
			tenncor::Int32Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& vec = arr.data();
			std::vector<int16_t> conv(vec.begin(), vec.end());
			*place = std::vector<uint16_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::UINT16:
		{
			tenncor::Uint32Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& vec = arr.data();
			std::vector<uint16_t> conv(vec.begin(), vec.end());
			*place = std::vector<uint16_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::INT32:
		{
			tenncor::Int32Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<int32_t>& vec = arr.data();
			*place = std::vector<int32_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::UINT32:
		{
			tenncor::Uint32Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint32_t>& vec = arr.data();
			*place = std::vector<uint32_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::INT64:
		{
			tenncor::Int64Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<int64_t>& vec = arr.data();
			*place = std::vector<int64_t>(vec.begin(), vec.end());
		}
		break;
		case tenncor::UINT64:
		{
			tenncor::Uint64Arr arr;
			tpb.data().UnpackTo(&arr);
			const google::protobuf::RepeatedField<uint64_t>& vec = arr.data();
			*place = std::vector<uint64_t>(vec.begin(), vec.end());
		}
		break;
		default:
			throw clay::UnsupportedTypeError((clay::DTYPE) tpb.type());
	}
}

std::unique_ptr<wire::Graph> load_graph (LeafSetT& leafset, RootIds& rootids,
	const tenncor::GraphPb& ingraph, const tenncor::DataRepoPb& in)
{
	std::string gid = ingraph.gid();
	assert(in.gid() == gid);
	std::unique_ptr<wire::Graph> graph = wire::Graph::get_temp(gid);

	const google::protobuf::Map<std::string,tenncor::NodePb>& nmap =
		ingraph.node_map();
	const google::protobuf::RepeatedPtrField<std::string>& order =
		ingraph.create_order();
	const google::protobuf::Map<std::string,tenncor::TensorPb>& dmap =
		in.data_map();
	for (std::string id : order)
	{
		wire::Identifier* node_id;
		const tenncor::NodePb& node = nmap.at(id);
		std::string label = node.label();
		switch (node.type())
		{
			case tenncor::NodePb::FUNCTOR:
			{
				tenncor::FunctorPb fpb;
				node.detail().UnpackTo(&fpb);
				const google::protobuf::RepeatedPtrField<std::string>&
                    arg_ids = fpb.args();
				std::vector<wire::Identifier*> args(arg_ids.size());
				std::transform(arg_ids.begin(), arg_ids.end(), args.begin(),
				[&](std::string argid)
				{
                    rootids.erase(argid);
					return graph->get_node(argid);
				});
				node_id = new wire::Functor(args, (slip::OPCODE) fpb.opcode(), *graph);
			}
			break;
			case tenncor::NodePb::CONSTANT:
			{
				tenncor::TensorPb tpb;
				node.detail().UnpackTo(&tpb);

				auto slist = tpb.shape();
				clay::Shape shape(std::vector<size_t>(slist.begin(), slist.end()));
				tenncor::TensorT type = tpb.type();
				std::shared_ptr<char> data = unpack_data(tpb.data(), type);
				node_id = new wire::Constant(data, shape, (clay::DTYPE) type, label, *graph);
				leafset.emplace(node_id);
			}
			break;
			case tenncor::NodePb::VARIABLE:
			{
				const tenncor::TensorPb& tpb = dmap.at(id);
				auto slist = tpb.shape();
				clay::Shape shape(std::vector<size_t>(slist.begin(), slist.end()));
				PbBuilder builder(tpb);
				node_id = new wire::Variable(builder, shape, label, *graph);
				leafset.emplace(node_id);
			}
			break;
			case tenncor::NodePb::PLACEHOLDER:
			{
				const tenncor::TensorPb& tpb = dmap.at(id);
				auto slist = tpb.shape();
				clay::Shape shape(std::vector<size_t>(slist.begin(), slist.end()));
				wire::Placeholder* place = new wire::Placeholder(shape, label, *graph);
				node_id = place;
				placeholder_assign(place, tpb);
				leafset.emplace(place);
			}
			break;
            default:
                throw std::exception(); // todo: add context
		}
		graph->replace_id(node_id, id);
		rootids.emplace(id);
	}

	return graph;
}

}

#endif
