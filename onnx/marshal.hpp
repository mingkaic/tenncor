#include "teq/ifunctor.hpp"

#include "marsh/objs.hpp"

#ifndef ONNX_MARSHAL_HPP
#define ONNX_MARSHAL_HPP

#include "onnx/onnx.pb.h"

namespace onnx
{

using PbAttrsT = ::google::protobuf::RepeatedPtrField<AttributeProto>;

const std::string leafconst_key = "immutable_leaf";

void marshal_attrs (PbAttrsT& out, const teq::iFunctor* func);

void unmarshal_attrs (marsh::Maps& out, const PbAttrsT& pb_attrs);

}

#endif // ONNX_MARSHAL_HPP
