///
/// serialize.hpp
/// tenncor
///
/// Purpose:
/// Define functions for marshal and unmarshal data sources
///

#include "internal/onnx/onnx.hpp"

#include "tenncor/layr/layr.hpp"

#ifndef TENNCOR_SERIAL_SERIALIZE_HPP
#define TENNCOR_SERIAL_SERIALIZE_HPP

namespace serial
{

template <typename CAST, typename T>
static inline void pack (char* data, size_t n,
	onnx::TensorProto& out, void(onnx::TensorProto::* insert)(T))
{
	CAST* ptr = (CAST*) data;
	for (size_t i = 0; i < n; ++i)
	{
		(out.*insert)(ptr[i]);
	}
}

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
		auto type_code = (egen::_GENERATED_DTYPE) tens.get_meta().type_code();
		auto code_name = egen::name_type(type_code);
		return name2onnxtype.at(code_name);
	}

	void marsh_leaf (onnx::TensorProto& out, const teq::iLeaf& leaf) const override
	{
		char* data = (char*) leaf.device().data();
		size_t nelems = leaf.shape().n_elems();
		auto type_code = (egen::_GENERATED_DTYPE) leaf.get_meta().type_code();
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
			case onnx::TensorProto::INT32:
				pack<int32_t>(data, nelems, out,
					&onnx::TensorProto::add_int32_data);
				break;
#ifdef EGEN_FULLTYPE
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
			case onnx::TensorProto::UINT64:
				pack<uint64_t>(data, nelems, out,
					&onnx::TensorProto::add_uint64_data);
				break;
			case onnx::TensorProto::INT64:
				pack<int64_t>(data, nelems, out,
					&onnx::TensorProto::add_int64_data);
				break;
#endif // EGEN_FULLTYPE
			default:
				global::fatalf("unknown onnx type %d (aka %s)",
					onnx_type, code_name.c_str());
		}
	}
};

template <typename TS> // todo: use tensor_range concept
void save_graph (onnx::GraphProto& pb_graph,
	const TS& roots, const onnx::TensIdT& identified = {},
	const teq::TensSetT& stops = {})
{
	MarshFuncs funcs;
	onnx::save_graph(pb_graph, roots, funcs, identified, stops);
}

teq::TensptrsT load_graph (onnx::TensptrIdT& identified_tens,
	const onnx::GraphProto& pb_graph);

}

#endif // HONE_SERIALIZE_HPP
