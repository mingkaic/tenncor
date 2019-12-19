#include <boost/bimap.hpp>

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"

#include "marsh/objs.hpp"

#ifndef ONNX_MARSHAL_HPP
#define ONNX_MARSHAL_HPP

#include "onnx/onnx.pb.h"

namespace onnx
{

using TensIdT = boost::bimap<teq::iTensor*,std::string>;

using TensptrIdT = boost::bimap<teq::TensptrT,std::string>;

using PbAttrsT = ::google::protobuf::RepeatedPtrField<AttributeProto>;

using TensorT = std::pair<int32_t,teq::Shape>;

using AnnotationsT = std::unordered_map<std::string,std::string>;

struct OnnxAttrMarshaler final : public marsh::iMarshaler
{
	OnnxAttrMarshaler (AttributeProto* out) : out_(out) {}

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

	AttributeProto* out_;
};

const std::string id_prelim = "::";

const std::string leafname_key = "TENSOR_NAME";

const std::string leafusage_key = "LEAF_USAGE";

const std::string subgraph_key = "LAYER_SUBGRAPH";

void marshal_attrs (PbAttrsT& out, const marsh::iAttributed& attrib);

void marshal_tensorshape (TensorShapeProto& out,
	const teq::Shape& shape);

void marshal_io (ValueInfoProto& out, const teq::Shape& shape);

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf);

const GraphProto* unmarshal_attrs (marsh::Maps& out, const PbAttrsT& pb_attrs);

teq::Shape unmarshal_shape (const TensorShapeProto& shape);

teq::Shape unmarshal_shape (const TensorProto& tens);

std::unordered_map<std::string,TensorT> unmarshal_io (
	const google::protobuf::RepeatedPtrField<ValueInfoProto>& values);

std::unordered_map<std::string,AnnotationsT> unmarshal_annotation (
	const google::protobuf::RepeatedPtrField<TensorAnnotation>& as);

}

#endif // ONNX_MARSHAL_HPP
