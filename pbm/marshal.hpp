#include "teq/shape.hpp"

#include "marsh/objs.hpp"

#ifndef PBM_MARSHAL_HPP
#define PBM_MARSHAL_HPP

#include "pbm/graph.pb.h"

namespace pbm
{

using PbAttrMapT = ::google::protobuf::Map<std::string,tenncor::ArrayAttrs>;

void marshal_attrs (PbAttrMapT& out, const marsh::Maps& attrs);

void unmarshal_attrs (marsh::Maps& out, const PbAttrMapT& pb_map);

teq::Shape get_shape (const tenncor::Source& source);

}

#endif // PBM_MARSHAL_HPP
