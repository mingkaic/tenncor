#include "eteq/serialize.hpp"

#ifdef ETEQ_SERIALIZE_HPP

namespace eteq
{

template <typename CAST, typename T>
static inline void pack (char* data, size_t n,
	onnx::TensorProto& out,
	void(onnx::TensorProto::* insert)(T))
{
	CAST* ptr = (CAST*) data;
	for (size_t i = 0; i < n; ++i)
	{
		(out.*insert)(ptr[i]);
	}
}

template <typename CAST, typename T>
static inline teq::TensptrT unpack (teq::Usage usage, teq::Shape shape,
	std::string label, const google::protobuf::RepeatedField<T>& data)
{
	std::vector<CAST> cdata(data.begin(), data.end());
	CAST* ptr = cdata.data();
	teq::TensptrT out;
	switch (usage) {
	case teq::Immutable:
		out = teq::TensptrT(Constant<CAST>::get(ptr, shape));
		break;
	case teq::Variable:
		out = teq::TensptrT(Variable<CAST>::get(ptr, shape, label, usage));
		break;
	case teq::Placeholder:
	{
		std::vector<CAST> z(shape.n_elems(), 0);
		out = teq::TensptrT(Variable<CAST>::get(z.data(), shape, label, usage));
	}
		break;
	default:
		logs::fatal("cannot unpack leaf of unknown usage");
	}
	return out;
}

template <typename T>
static inline teq::TensptrT convert_func (std::string opname,
	const teq::TensptrsT& children, marsh::Maps&& attrs)
{
	return Functor<T>::get(egen::get_op(opname), children, std::move(attrs));
}

#define _OUT_GENFUNC(realtype)\
func = Functor<realtype>::get(opcode, children, std::move(attrs));

#define _OUT_GENLAYR(realtype)\
layer = teq::TensptrT(Layer<realtype>::get(opcode, children, root));

// todo: move this to generated layer
static const std::unordered_map<
	std::string,onnx::TensorProto::DataType> name2onnxtype = {
	{"DOUBLE", onnx::TensorProto::DOUBLE},
	{"FLOAT", onnx::TensorProto::FLOAT},
	{"UINT8", onnx::TensorProto::UINT8},
	{"INT8", onnx::TensorProto::INT8},
	{"UINT16", onnx::TensorProto::UINT16},
	{"INT16", onnx::TensorProto::INT16},
	{"UINT32", onnx::TensorProto::UINT32},
	{"INT32", onnx::TensorProto::INT32},
	{"UINT64", onnx::TensorProto::UINT64},
	{"INT64", onnx::TensorProto::INT64},
};

struct MarshFuncs final : public onnx::iMarshFuncs
{
	size_t get_typecode (const teq::iTensor& tens) const override
	{
		auto type_code = (egen::_GENERATED_DTYPE) tens.type_code();
		auto code_name = egen::name_type(type_code);
		return name2onnxtype.at(code_name);
	}

	void marsh_leaf (onnx::TensorProto& out, const teq::iLeaf& leaf) const override
	{
		char* data = (char*) leaf.data();
		size_t nelems = leaf.shape().n_elems();
		auto type_code = (egen::_GENERATED_DTYPE) leaf.type_code();
		auto code_name = egen::name_type(type_code);
		auto onnx_type = name2onnxtype.at(code_name);
		out.set_data_type(onnx_type);
		switch (onnx_type)
		{
			case onnx::TensorProto::DOUBLE:
				pack<double>(data, nelems, out,
					&onnx::TensorProto::add_double_data);
				break;
			case onnx::TensorProto::FLOAT:
				pack<float>(data, nelems, out,
					&onnx::TensorProto::add_float_data);
				break;
			case onnx::TensorProto::UINT8:
				pack<uint8_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
			case onnx::TensorProto::INT8:
				pack<int8_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
			case onnx::TensorProto::UINT16:
				pack<uint16_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
			case onnx::TensorProto::INT16:
				pack<int16_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
			case onnx::TensorProto::UINT32:
				pack<uint32_t>(data, nelems, out,
					&onnx::TensorProto::add_uint64_data);
				break;
			case onnx::TensorProto::INT32:
				pack<int32_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
			case onnx::TensorProto::UINT64:
				pack<uint64_t>(data, nelems, out,
					&onnx::TensorProto::add_uint64_data);
				break;
			case onnx::TensorProto::INT64:
				pack<int64_t>(data, nelems, out,
					&onnx::TensorProto::add_int64_data);
				break;
			default:
				logs::fatalf("unknown onnx type %d (aka %s)",
					onnx_type, code_name.c_str());
		}
	}
};

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
				logs::fatalf("unknown onnx type %d", onnx_type);
		}
		return out;
	}

	teq::TensptrT unmarsh_func (std::string opname,
		const teq::TensptrsT& children, marsh::Maps&& attrs) const override
	{
		if (children.empty())
		{
			logs::fatalf("cannot generate func %s without args", opname.c_str());
		}
		egen::_GENERATED_OPCODE opcode = egen::get_op(opname);
		size_t gencode = children.front()->type_code();
		teq::TensptrT func = nullptr;
		TYPE_LOOKUP(_OUT_GENFUNC, (egen::_GENERATED_DTYPE) gencode);
		return func;
	}

	teq::TensptrT unmarsh_layr (std::string opname,
		const teq::TensptrsT& roots, const teq::TensptrsT& children,
		marsh::Maps&& attrs) const override
	{
		if (roots.empty())
		{
			logs::fatal("cannot unmarshal layr without any roots");
		}
		teq::FuncptrT root = std::static_pointer_cast<teq::iFunctor>(roots.front());
		size_t gencode = root->type_code();
		teq::Opcode opcode{opname, 0};
		teq::TensptrT layer = nullptr;
		TYPE_LOOKUP(_OUT_GENLAYR, (egen::_GENERATED_DTYPE) gencode);
		return layer;
	}
};

#undef _OUT_GENFUNC

#undef _OUT_GENLAYR

void save_graph (onnx::GraphProto& pb_graph, teq::TensptrsT roots,
	const onnx::TensIdT& identified)
{
	MarshFuncs funcs;
	onnx::save_graph(pb_graph, roots, funcs, identified);
}

teq::TensptrsT load_graph (onnx::TensptrIdT& identified_tens,
	const onnx::GraphProto& pb_graph)
{
	UnmarshFuncs funcs;
	return onnx::load_graph(identified_tens, pb_graph, funcs);
}

}

#endif
