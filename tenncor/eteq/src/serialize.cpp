
#include "eteq/serialize.hpp"

#ifdef ETEQ_SERIALIZE_HPP

namespace eteq
{

template <typename CAST, typename T>
static inline teq::TensptrT unpack (teq::Usage usage, teq::Shape shape,
	std::string label, const google::protobuf::RepeatedField<T>& data)
{
	std::vector<CAST> cdata(data.begin(), data.end());
	CAST* ptr = cdata.data();
	teq::TensptrT out;
	switch (usage) {
	case teq::IMMUTABLE:
		out = teq::TensptrT(Constant<CAST>::get(ptr, shape));
		break;
	case teq::VARUSAGE:
		out = teq::TensptrT(Variable<CAST>::get(ptr, shape, label, usage));
		break;
	case teq::PLACEHOLDER:
	{
		std::vector<CAST> z(shape.n_elems(), 0);
		out = teq::TensptrT(Variable<CAST>::get(z.data(), shape, label, usage));
	}
		break;
	default:
		teq::fatal("cannot unpack leaf of unknown usage");
	}
	return out;
}

#define _OUT_GENFUNC(realtype)\
func = make_funcattr<realtype>(opcode, children, attrs);

struct UnmarshFuncs final : public onnx::iUnmarshFuncs
{
	teq::TensptrT unmarsh_leaf (const onnx::TensorProto& pb_tens,
		teq::Usage usage, std::string label) const override
	{
		teq::TensptrT out;
		teq::Shape shape = onnx::unmarshal_shape(pb_tens);
		auto onnx_type = pb_tens.data_type();
		switch (onnx_type)
		{
			case onnx::TensorProto::DOUBLE:
				out = unpack<double>(usage, shape, label,
					pb_tens.double_data());
				break;
			case onnx::TensorProto::FLOAT:
				out = unpack<float>(usage, shape, label,
					pb_tens.float_data());
				break;
			case onnx::TensorProto::INT32:
				out = unpack<int32_t>(usage, shape, label,
					pb_tens.int32_data());
				break;
// #if(ETEQ_CFG==FULL)
// 			case onnx::TensorProto::UINT8:
// 				out = unpack<uint8_t>(usage, shape, label,
// 					pb_tens.int32_data());
// 				break;
// 			case onnx::TensorProto::INT8:
// 				out = unpack<int8_t>(usage, shape, label,
// 					pb_tens.int32_data());
// 				break;
// 			case onnx::TensorProto::UINT16:
// 				out = unpack<uint16_t>(usage, shape, label,
// 					pb_tens.int32_data());
// 				break;
// 			case onnx::TensorProto::INT16:
// 				out = unpack<int16_t>(usage, shape, label,
// 					pb_tens.int32_data());
// 				break;
// 			case onnx::TensorProto::UINT32:
// 				out = unpack<uint32_t>(usage, shape, label,
// 					pb_tens.uint64_data());
// 				break;
// 			case onnx::TensorProto::UINT64:
// 				out = unpack<uint64_t>(usage, shape, label,
// 					pb_tens.uint64_data());
// 				break;
// 			case onnx::TensorProto::INT64:
// 				out = unpack<int64_t>(usage, shape, label,
// 					pb_tens.int64_data());
// 				break;
// #endif
			default:
				teq::fatalf("unknown onnx type %d", onnx_type);
		}
		return out;
	}

	teq::TensptrT unmarsh_func (std::string opname,
		const teq::TensptrsT& children, marsh::Maps&& attrs) const override
	{
		if (children.empty())
		{
			teq::fatalf("cannot generate func %s without args", opname.c_str());
		}
		egen::_GENERATED_OPCODE opcode = egen::get_op(opname);
		auto gencode = (egen::_GENERATED_DTYPE) children.front()->get_meta().type_code();
		teq::TensptrT func;
		TYPE_LOOKUP(_OUT_GENFUNC, gencode);
		return func;
	}

	teq::TensptrT unmarsh_layr (std::string layername,
		const teq::TensptrT& root, const teq::TensptrT& child,
		marsh::Maps&& attrs) const override
	{
		return eteq::make_layer(root, layername, child);
	}
};

#undef _OUT_GENFUNC

teq::TensptrsT load_model (onnx::TensptrIdT& identified_tens,
	const onnx::ModelProto& pb_model)
{
	UnmarshFuncs funcs;
	return onnx::load_graph(identified_tens, pb_model.graph(), funcs);
}

}

#endif
