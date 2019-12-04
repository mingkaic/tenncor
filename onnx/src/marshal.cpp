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
			out.contents_.emplace(pb_attr.name(), marsh::ObjptrT(out_arr));
		}
	}
}

}

#endif
