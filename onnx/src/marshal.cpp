#include "onnx/marshal.hpp"

#ifdef ONNX_MARSHAL_HPP

namespace onnx
{

void marshal_attrs (PbAttrsT& out, const teq::iFunctor* func)
{
	std::vector<std::string> attr_keys = func->ls_attrs();
	for (std::string attr_key : attr_keys)
	{
		auto attr = func->get_attr(attr_key);
		if (typeid(marsh::NumArray<double>).
			hash_code() != attr->class_code())
		{
			continue;
		}
		auto& contents = static_cast<
			const marsh::NumArray<double>*>(attr)->contents_;
		AttributeProto* pb_attrs = out.Add();
		pb_attrs->set_name(attr_key);
		pb_attrs->set_type(AttributeProto::FLOATS);
		google::protobuf::RepeatedField<float> tmp(contents.begin(), contents.end());
		pb_attrs->mutable_floats()->Swap(&tmp);
	}
}

void marshal_tensorshape (TensorShapeProto& out,
	const teq::ShapeSignature& shape)
{
	for (teq::DimT dim : shape)
	{
		out.add_dim()->set_dim_value(dim);
	}
}

void marshal_io (ValueInfoProto& out, int32_t data_type,
	const teq::ShapeSignature& shape)
{
	TypeProto* type = out.mutable_type();
	TypeProto::Tensor* tens_type = type->mutable_tensor_type();
	tens_type->set_elem_type(data_type);
	marshal_tensorshape(*tens_type->mutable_shape(), shape);
}

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf)
{
	StringStringEntryProto* namer = out.add_quant_parameter_tensor_names();
	namer->set_key(leafname_key);
	namer->set_value(leaf.to_string());
	if (leaf.is_const())
	{
		StringStringEntryProto* immuter =
			out.add_quant_parameter_tensor_names();
		immuter->set_key(leafconst_key);
	}
}

void unmarshal_attrs (marsh::Maps& out, const PbAttrsT& pb_attrs)
{
	for (const auto& pb_attr : pb_attrs)
	{
		if (pb_attr.type() == AttributeProto::FLOATS)
		{
			auto& pb_values = pb_attr.floats();
			auto out_arr = new marsh::NumArray<double>();
			for (float e : pb_values)
			{
				out_arr->contents_.push_back(e);
			}
			out.add_attr(pb_attr.name(), marsh::ObjptrT(out_arr));
		}
	}
}

teq::Shape unmarshal_shape (const TensorShapeProto& shape)
{
	const auto& dims = shape.dim();
	std::vector<teq::DimT> slist;
	slist.reserve(dims.size());
	std::transform(dims.begin(), dims.end(), std::back_inserter(slist),
		[](const TensorShapeProto::Dimension& dim)
		{
			assert(dim.has_dim_value());
			return dim.dim_value();
		});
	return teq::Shape(slist);
}

teq::Shape unmarshal_shape (const TensorProto& tens)
{
	auto dims = tens.dims();
	std::vector<teq::DimT> slist(dims.begin(), dims.end());
	return teq::Shape(slist);
}

std::unordered_map<std::string,TensorT> unmarshal_io (
	const google::protobuf::RepeatedPtrField<ValueInfoProto>& values)
{
	std::unordered_map<std::string,TensorT> out;
	for (const ValueInfoProto& value : values)
	{
		std::string id = value.name();
		const TypeProto& type = value.type();
		assert(type.has_tensor_type());
		const TypeProto::Tensor& tens_type = type.tensor_type();
		out.emplace(id, TensorT{tens_type.elem_type(),
			unmarshal_shape(tens_type.shape())});
	}
	return out;
}

std::unordered_map<std::string,AnnotationsT> unmarshal_annotation (
	const google::protobuf::RepeatedPtrField<TensorAnnotation>& as)
{
	std::unordered_map<std::string,AnnotationsT> out;
	for (const TensorAnnotation& annotation : as)
	{
		std::string id = annotation.tensor_name();
		const auto& params = annotation.quant_parameter_tensor_names();
		for (const StringStringEntryProto& param : params)
		{
			out[id].emplace(param.key(), param.value());
		}
	}
	return out;
}

}

#endif
