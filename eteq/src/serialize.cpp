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
static inline teq::TensptrT unpack (bool is_const, teq::Shape shape,
	std::string label, const google::protobuf::RepeatedField<T>& data)
{
	std::vector<CAST> cdata(data.begin(), data.end());
	CAST* ptr = cdata.data();
	if (is_const)
	{
		return teq::TensptrT(Constant<CAST>::get(ptr, shape));
	}
	return teq::TensptrT(Variable<CAST>::get(ptr, shape, label));
}

static inline std::vector<double> convert_attrs (
	const marsh::Maps& attrs, std::string key)
{
	std::vector<double> out;
	if (estd::has(attrs.contents_, key))
	{
		const auto& objs = attrs.contents_.at(key);
		if (typeid(marsh::NumArray<double>).hash_code() == objs->class_code())
		{
			out = static_cast<const marsh::NumArray<double>*>(
				objs.get())->contents_;
		}
	}
	return out;
}

template <typename T>
static inline teq::TensptrT convert_func (std::string opname,
	const teq::TensptrsT& args, marsh::Maps&& attrs)
{
	ArgsT<T> edges;
	edges.reserve(args.size());
	std::transform(args.begin(), args.end(), std::back_inserter(edges),
		[](teq::TensptrT tens)
		{
			return Edge<T>(to_node<T>(tens));
		});
	return teq::TensptrT(Functor<T>::get(
		egen::get_op(opname), edges, std::move(attrs)));
}

#define _OUT_GENFUNC(realtype)\
func = convert_func<realtype>(opname, args, std::move(attrs));

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

static void save_leaf (onnx::TensorProto& out, const teq::iLeaf& leaf)
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
			logs::fatalf("unknown onnx type %d", onnx_type);
	}
}

teq::TensptrT load_leaf (const onnx::TensorProto& pb_tens,
	bool is_const, std::string label)
{
	teq::TensptrT out;
	teq::Shape shape = onnx::unmarshal_shape(pb_tens);
	auto onnx_type = pb_tens.data_type();
	switch (onnx_type)
	{
		case onnx::TensorProto::DOUBLE:
			out = unpack<double>(is_const, shape, label,
				pb_tens.double_data());
			break;
		case onnx::TensorProto::FLOAT:
			out = unpack<float>(is_const, shape, label,
				pb_tens.float_data());
			break;
		case onnx::TensorProto::INT32:
			out = unpack<int32_t>(is_const, shape, label,
				pb_tens.int32_data());
			break;
// #if(ETEQ_CFG==FULL)
// 		case onnx::TensorProto::UINT8:
// 			out = unpack<uint8_t>(is_const, shape, label,
// 				pb_tens.int32_data());
// 			break;
// 		case onnx::TensorProto::INT8:
// 			out = unpack<int8_t>(is_const, shape, label,
// 				pb_tens.int32_data());
// 			break;
// 		case onnx::TensorProto::UINT16:
// 			out = unpack<uint16_t>(is_const, shape, label,
// 				pb_tens.int32_data());
// 			break;
// 		case onnx::TensorProto::INT16:
// 			out = unpack<int16_t>(is_const, shape, label,
// 				pb_tens.int32_data());
// 			break;
// 		case onnx::TensorProto::UINT32:
// 			out = unpack<uint32_t>(is_const, shape, label,
// 				pb_tens.uint64_data());
// 			break;
// 		case onnx::TensorProto::UINT64:
// 			out = unpack<uint64_t>(is_const, shape, label,
// 				pb_tens.uint64_data());
// 			break;
// 		case onnx::TensorProto::INT64:
// 			out = unpack<int64_t>(is_const, shape, label,
// 				pb_tens.int64_data());
// 			break;
// #endif
		default:
			logs::fatalf("unknown onnx type %d", onnx_type);
	}
	return out;
}

teq::TensptrT load_func (std::string opname,
	const teq::TensptrsT& args, marsh::Maps&& attrs)
{
	if (args.empty())
	{
		logs::fatalf("cannot generate func %s without args", opname.c_str());
	}
	size_t gencode = egen::BAD_TYPE;
	auto ctens = args[0].get();
	if (auto leaf = dynamic_cast<teq::iLeaf*>(ctens))
	{
		gencode = leaf->type_code();
	}
	else if (auto func = dynamic_cast<teq::iOperableFunc*>(ctens))
	{
		gencode = func->type_code();
	}
	else
	{
		logs::fatalf("cannot generate func from non-eteq tensor arg %s",
			ctens->to_string().c_str());
	}
	teq::TensptrT func = nullptr;
	TYPE_LOOKUP(_OUT_GENFUNC, (egen::_GENERATED_DTYPE) gencode);
	return func;
}

#undef _OUT_GENFUNC

void save_graph (onnx::GraphProto& pb_graph, teq::TensptrsT roots)
{
	return onnx::save_graph(pb_graph, roots, save_leaf);
}

void load_graph (teq::TensptrsT& roots, const onnx::GraphProto& pb_graph)
{
	return onnx::load_graph(roots, pb_graph, load_leaf, load_func);
}

static bool is_big_endian(void)
{
	union
	{
		uint16_t _;
		char bytes[2];
	} twob = { 0x0001 };

	return twob.bytes[0] == 0;
}

template <typename T>
static inline teq::TensptrT pbm_convert_func (std::string opname,
	const pbm::EdgesT& edges, marsh::Maps&& attrs)
{
	ArgsT<T> tmp_edges;
	tmp_edges.reserve(edges.size());
	for (auto& edge : edges)
	{
		teq::Shape shape = edge.first->shape();
		auto shape_vals = convert_attrs(edge.second, eigen::shaper_key);
		if (shape_vals.size() > 0)
		{
			shape = teq::Shape(std::vector<teq::DimT>(
				shape_vals.begin(), shape_vals.end()));
		}
		tmp_edges.push_back(
			Edge<T>(to_node<T>(edge.first)));
	}
	return std::shared_ptr<Functor<T>>(Functor<T>::get(
		egen::get_op(opname), tmp_edges, std::move(attrs)));
}

static std::string marshal_leaf (teq::iLeaf* leaf)
{
	char* data = (char*) leaf->data();
	size_t nelems = leaf->shape().n_elems();
	size_t nbytes = egen::type_size(
		(egen::_GENERATED_DTYPE) leaf->type_code());
	if (is_big_endian() && nbytes > 1)
	{
		size_t totalbytes = nelems * nbytes;
		std::string out(totalbytes, '\0');
		for (size_t i = 0; i < totalbytes; ++i)
		{
			size_t elemi = i / nbytes;
			size_t outi = (elemi + 1) * nbytes - (i % nbytes);
			out[outi] = data[i];
		}
		return out;
	}
	return std::string(data, nelems * nbytes);
}

pbm::TensMapIndicesT save_graph (
	tenncor::Graph& out, teq::TensptrsT roots,
	tag::TagRegistry& registry)
{
	return pbm::save_graph(out, roots, registry, marshal_leaf);
}

#define _OUT_GENERIC(realtype)leaf = is_const?\
teq::TensptrT(Constant<realtype>::get((realtype*) pb, shape)):\
teq::TensptrT(Variable<realtype>::get((realtype*) pb, shape, label));

static teq::TensptrT unmarshal_leaf (
	const tenncor::Source& source, std::string label)
{
	teq::Shape shape = pbm::get_shape(source);
	const char* pb = source.data().c_str();
	bool is_const = source.is_const();

	teq::TensptrT leaf;
	egen::_GENERATED_DTYPE gencode = egen::get_type(source.typelabel());
	size_t nbytes = egen::type_size(gencode);
	if (is_big_endian() && nbytes > 1)
	{
		size_t totalbytes = shape.n_elems() * nbytes;
		std::string out(totalbytes, '\0');
		for (size_t i = 0; i < totalbytes; ++i)
		{
			size_t elemi = i / nbytes;
			size_t outi = (elemi + 1) * nbytes - (i % nbytes);
			out[outi] = pb[i];
		}
		pb = out.c_str();
		TYPE_LOOKUP(_OUT_GENERIC, gencode)
	}
	else
	{
		TYPE_LOOKUP(_OUT_GENERIC, gencode)
	}
	return leaf;
}

#undef _OUT_GENERIC

#define _OUT_GENFUNC(realtype)\
func = pbm_convert_func<realtype>(opname, edges, std::move(attrs));

static teq::TensptrT unmarshal_func (std::string opname,
	const pbm::EdgesT& edges, marsh::Maps&& attrs)
{
	if (edges.empty())
	{
		logs::fatalf("cannot generate func %s without edges", opname.c_str());
	}
	size_t gencode = egen::BAD_TYPE;
	auto ctens = edges[0].first.get();
	if (auto leaf = dynamic_cast<teq::iLeaf*>(ctens))
	{
		gencode = leaf->type_code();
	}
	else if (auto func = dynamic_cast<teq::iOperableFunc*>(ctens))
	{
		gencode = func->type_code();
	}
	else
	{
		logs::fatalf("cannot generate func from non-eteq tensor arg %s",
			ctens->to_string().c_str());
	}
	teq::TensptrT func = nullptr;
	TYPE_LOOKUP(_OUT_GENFUNC, (egen::_GENERATED_DTYPE) gencode);
	return func;
}

#undef _OUT_GENFUNC

void load_graph (teq::TensptrSetT& roots,
	const tenncor::Graph& pb_graph,
	tag::TagRegistry& registry)
{
	pbm::load_graph(roots, pb_graph, registry, unmarshal_leaf, unmarshal_func);
}

}

#endif
