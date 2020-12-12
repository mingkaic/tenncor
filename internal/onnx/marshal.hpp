
#ifndef ONNX_MARSHAL_HPP
#define ONNX_MARSHAL_HPP

#include <boost/bimap.hpp>

#include "internal/teq/teq.hpp"

#include "internal/onnx/onnx.pb.h"

namespace onnx
{

using TensIdT = boost::bimap<teq::iTensor*,std::string>;

using TensptrIdT = boost::bimap<teq::TensptrT,std::string>;

using PbAttrsT = ::google::protobuf::RepeatedPtrField<AttributeProto>;

using TensorT = std::pair<int32_t,teq::Shape>;

using AnnotationsT = types::StrUMapT<std::string>;

const std::string leafname_key = "TENSOR_NAME";

const std::string leafusage_key = "LEAF_USAGE";

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
		types::StringsT strs;
		std::vector<int64_t> ints;
		std::vector<double> floats;
		types::StringsT tensors;
		arr.foreach(
			[&](size_t, const marsh::iObject* obj)
			{
				if (auto str = dynamic_cast<const marsh::String*>(obj))
				{
					strs.push_back(str->to_string());
				}
				else if (auto num = dynamic_cast<const marsh::iNumber*>(obj))
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
				else if (auto tens = dynamic_cast<const teq::TensorObj*>(obj))
				{
					auto mtens = tens->get_tensor().get();
					auto id = estd::must_getf(tensid_, mtens,
						"cannot find %s", mtens->to_string().c_str());
					tensors.push_back(id);
				}
			});
		if (false == arr.is_primitive())
		{
			if (typeid(marsh::String).hash_code() == arr.subclass_code())
			{
				out_->set_type(AttributeProto::STRINGS);
				for (auto str : strs)
				{
					out_->add_strings(str);
				}
			}
			else
			{
				out_->set_type(AttributeProto::TENSORS);
				for (auto id : tensors)
				{
					auto tens = out_->add_tensors();
					tens->set_name(id);
				}
			}
		}
		else if (arr.is_integral())
		{
			out_->set_type(AttributeProto::INTS);
			for (auto num : ints)
			{
				out_->add_ints(num);
			}
		}
		else
		{
			out_->set_type(AttributeProto::FLOATS);
			for (auto num : floats)
			{
				out_->add_floats(num);
			}
		}
	}

	void marshal (const marsh::iTuple&) override
	{
		global::fatal("onnx does not support tuple attributes");
	}

	void marshal (const marsh::Maps&) override
	{
		global::fatal("onnx does not support map attributes");
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

	void marshal (const teq::LayerObj&) override
	{
		// ignore: since marshaler resolves graph info
		global::fatal("onnx does not support layer attributes");
	}

	AttributeProto* out_;

	const teq::CTensMapT<std::string>& tensid_;
};

void marshal_attrs (PbAttrsT& out, const marsh::iAttributed& attrib,
	const teq::CTensMapT<std::string>& tensid);

void marshal_tensorshape (TensorShapeProto& out,
	const teq::Shape& shape);

void marshal_io (ValueInfoProto& out, const teq::Shape& shape);

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf);

const GraphProto* unmarshal_attrs (marsh::Maps& out,
	const PbAttrsT& pb_attrs, const TensptrIdT& identified_tens);

teq::Shape unmarshal_shape (const TensorProto& tens);

types::StrUMapT<AnnotationsT> unmarshal_annotation (
	const google::protobuf::RepeatedPtrField<TensorAnnotation>& as);

}

#endif // ONNX_MARSHAL_HPP
