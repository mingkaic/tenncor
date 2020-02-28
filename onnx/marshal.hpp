
#ifndef ONNX_MARSHAL_HPP
#define ONNX_MARSHAL_HPP

#include <boost/bimap.hpp>

#include "teq/ileaf.hpp"
#include "teq/ifunctor.hpp"
#include "teq/objs.hpp"

#include "onnx/onnx.pb.h"

namespace onnx
{

using TensIdT = boost::bimap<teq::iTensor*,std::string>;

using TensptrIdT = boost::bimap<teq::TensptrT,std::string>;

using PbAttrsT = ::google::protobuf::RepeatedPtrField<AttributeProto>;

using TensorT = std::pair<int32_t,teq::Shape>;

using AnnotationsT = std::unordered_map<std::string,std::string>;

const std::string leafname_key = "TENSOR_NAME";

const std::string leafusage_key = "LEAF_USAGE";

void marshal_attrs (PbAttrsT& out, const marsh::iAttributed& attrib,
	const teq::CTensMapT<std::string>& tensid);

void marshal_tensorshape (TensorShapeProto& out,
	const teq::Shape& shape);

void marshal_io (ValueInfoProto& out, const teq::Shape& shape);

void marshal_annotation (TensorAnnotation& out, const teq::iLeaf& leaf);

const GraphProto* unmarshal_attrs (marsh::Maps& out,
	const PbAttrsT& pb_attrs, const TensptrIdT& identified_tens);

teq::Shape unmarshal_shape (const TensorShapeProto& shape);

teq::Shape unmarshal_shape (const TensorProto& tens);

std::unordered_map<std::string,TensorT> unmarshal_io (
	const google::protobuf::RepeatedPtrField<ValueInfoProto>& values);

std::unordered_map<std::string,AnnotationsT> unmarshal_annotation (
	const google::protobuf::RepeatedPtrField<TensorAnnotation>& as);

}

#endif // ONNX_MARSHAL_HPP
