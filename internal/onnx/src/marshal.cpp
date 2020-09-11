#include "internal/onnx/marshal.hpp"

#ifdef ONNX_MARSHAL_HPP

namespace onnx
{

void marshal_attrs (PbAttrsT& out, const marsh::iAttributed& attrib,
	const teq::CTensMapT<std::string>& tensid)
{
	types::StringsT attr_keys = attrib.ls_attrs();
	for (const std::string& attr_key : attr_keys)
	{
		if (attr_key == teq::layer_attr)
		{
			continue; // ignore: since marshaler resolves graph info
		}
		auto attr = attrib.get_attr(attr_key);
		AttributeProto* pb_attrs = out.Add();
		pb_attrs->set_name(attr_key);

		OnnxAttrMarshaler marsh(pb_attrs, tensid);
		attr->accept(marsh);
	}
}

void marshal_tensorshape (TensorShapeProto& out,
	const teq::Shape& shape)
{
	for (teq::DimT dim : shape)
	{
		out.add_dim()->set_dim_value(dim);
	}
}

void marshal_io (ValueInfoProto& out, const teq::Shape& shape)
{
	TypeProto* type = out.mutable_type();
	TypeProto::Tensor* tens_type = type->mutable_tensor_type();
	marshal_tensorshape(*tens_type->mutable_shape(), shape);
}

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf)
{
	StringStringEntryProto* namer = out.add_quant_parameter_tensor_names();
	namer->set_key(leafname_key);
	namer->set_value(leaf.to_string());
	auto tenspair = out.add_quant_parameter_tensor_names();
	tenspair->set_key(leafusage_key);
	tenspair->set_value(teq::get_usage_name(leaf.get_usage()));
}

using StrArrayT = marsh::PtrArray<marsh::String>;

const GraphProto* unmarshal_attrs (marsh::Maps& out,
	const PbAttrsT& pb_attrs, const TensptrIdT& identified_tens)
{
	const GraphProto* subgraph = nullptr;
	for (const auto& pb_attr : pb_attrs)
	{
		std::string attr_name = pb_attr.name();
		marsh::iObject* val = nullptr;
		switch (pb_attr.type())
		{
			case AttributeProto::STRING:
				val = new marsh::String(pb_attr.s());
				break;
			case AttributeProto::INT:
				val = new marsh::Number<int64_t>(pb_attr.i());
				break;
			case AttributeProto::FLOAT:
				val = new marsh::Number<double>(pb_attr.f());
				break;
			case AttributeProto::STRINGS:
			{
				auto& pb_values = pb_attr.strings();
				auto strs = new StrArrayT();
				val = strs;
				auto& content = strs->contents_;
				for (const std::string& e : pb_values)
				{
					content.emplace(content.end(),
						std::make_unique<marsh::String>(e));
				}
			}
				break;
			case AttributeProto::INTS:
			{
				auto& pb_values = pb_attr.ints();
				auto ints = new marsh::NumArray<int64_t>();
				val = ints;
				for (auto e : pb_values)
				{
					ints->contents_.push_back(e);
				}
			}
				break;
			case AttributeProto::FLOATS:
			{
				auto& pb_values = pb_attr.floats();
				auto floats = new marsh::NumArray<double>();
				val = floats;
				for (float e : pb_values)
				{
					floats->contents_.push_back(e);
				}
			}
				break;
			case AttributeProto::TENSOR:
			{
				auto& pb_tens = pb_attr.t();
				std::string id = pb_tens.name();
				val = new teq::TensorObj(estd::must_getf(
					identified_tens.right, id,
					"cannot find tensor id %s", id.c_str()));
			}
				break;
			case AttributeProto::TENSORS:
			{
				auto& pb_values = pb_attr.tensors();
				auto tens = new marsh::PtrArray<teq::TensorObj>();
				val = tens;
				auto& content = tens->contents_;
				for (const auto& pb_tens : pb_values)
				{
					std::string id = pb_tens.name();
					content.emplace(content.end(),
						std::make_unique<teq::TensorObj>(
							estd::must_getf(identified_tens.right, id,
							"cannot find tensor id %s", id.c_str())));
				}
			}
				break;
			case AttributeProto::GRAPH:
				if (attr_name == teq::layer_attr)
				{
					subgraph = &pb_attr.g();
				}
				else
				{
					global::fatalf("unknown graph attribute `%s`",
						attr_name.c_str());
				}
				continue;
			default:
				global::fatalf("unknown onnx attribute type of `%s`",
					attr_name.c_str());
		}
		out.add_attr(attr_name, marsh::ObjptrT(val));
	}
	return subgraph;
}

teq::Shape unmarshal_shape (const TensorProto& tens)
{
	auto dims = tens.dims();
	teq::DimsT slist(dims.begin(), dims.end());
	return teq::Shape(slist);
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
