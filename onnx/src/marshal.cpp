#include "onnx/marshal.hpp"

#ifdef ONNX_MARSHAL_HPP

namespace onnx
{

struct OnnxAttrMarshaler final : public teq::iTeqMarshaler
{
	OnnxAttrMarshaler (AttributeProto* out,
		const teq::CTensMapT<std::string>& tensid) :
		out_(out), tensid_(tensid) {}

	void marshal (const marsh::String& str) override
	{
		out_->set_type(AttributeProto::STRING);
		out_->set_s(str.to_string());
	}

	void marshal (const marsh::iNumber& num) override
	{
		if (num.is_integral())
		{
			out_->set_type(AttributeProto::INT);
			out_->set_i(num.to_int64());
		}
		else
		{
			out_->set_type(AttributeProto::FLOAT);
			out_->set_f(num.to_float64());
		}
	}

	void marshal (const marsh::iArray& arr) override
	{
		std::vector<std::string> strs;
		std::vector<int64_t> ints;
		std::vector<double> floats;
		arr.foreach(
			[&](size_t i, const marsh::ObjptrT& obj)
			{
				if (auto str = dynamic_cast<const marsh::String*>(obj.get()))
				{
					strs.push_back(str->to_string());
				}
				else if (auto num = dynamic_cast<const marsh::iNumber*>(obj.get()))
				{
					if (num->is_integral())
					{
						ints.push_back(num->to_int64());
					}
					else
					{
						floats.push_back(num->to_float64());
					}
				}
			});
		if (strs.size() > 0 && ints.size() > 0 && floats.size() > 0)
		{
			logs::fatal("onnx does not support hetero-typed arrays");
		}
		if (strs.size() > 0)
		{
			out_->set_type(AttributeProto::STRINGS);
			for (auto str : strs)
			{
				out_->add_strings(str);
			}
		}
		else if (ints.size() > 0)
		{
			out_->set_type(AttributeProto::INTS);
			for (auto num : ints)
			{
				out_->add_ints(num);
			}
		}
		else
		{
			// by default, assume to be float array
			out_->set_type(AttributeProto::FLOATS);
			for (auto num : floats)
			{
				out_->add_floats(num);
			}
		}
	}

	void marshal (const marsh::Maps& mm) override
	{
		logs::fatal("onnx does not support map attributes");
	}

	void marshal (const teq::TensorObj& tens) override
	{
		auto mtens = tens.get_tensor().get();
		auto id = estd::must_getf(tensid_, mtens,
			"cannot find %s", mtens->to_string().c_str());
		out_->set_type(AttributeProto::TENSOR);
		auto pb_tens = out_->mutable_t();
		pb_tens->set_name(id);
	}

	void marshal (const teq::LayerObj& layer) override
	{
		// ignore: since marshaler resolves graph info
	}

	AttributeProto* out_;

	const teq::CTensMapT<std::string>& tensid_;
};

void marshal_attrs (PbAttrsT& out, const marsh::iAttributed& attrib,
	const teq::CTensMapT<std::string>& tensid)
{
	std::vector<std::string> attr_keys = attrib.ls_attrs();
	for (std::string attr_key : attr_keys)
	{
		if (attr_key == teq::layer_key)
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
				auto strs = new marsh::ObjArray();
				val = strs;
				auto& content = strs->contents_;
				for (std::string e : pb_values)
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
			case AttributeProto::GRAPH:
				if (attr_name == teq::layer_key)
				{
					subgraph = &pb_attr.g();
				}
				else
				{
					logs::warnf("unknown graph attribute %s",
						attr_name.c_str());
				}
				continue;
			default:
				logs::fatalf("unknown onnx attribute type of %s",
					attr_name.c_str());
		}
		out.add_attr(attr_name, marsh::ObjptrT(val));
	}
	return subgraph;
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
