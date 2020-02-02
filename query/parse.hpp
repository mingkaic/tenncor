//
/// parse.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering condition parsing
///

#ifndef QUERY_PARSE_HPP
#define QUERY_PARSE_HPP

#include "teq/ileaf.hpp"

#include "query/query.pb.h"

namespace query
{

using ConditionT = std::shared_ptr<Node>;

static inline teq::Shape to_shape (
	const google::protobuf::RepeatedField<uint32_t>& sfields)
{
	std::vector<teq::DimT> slist(sfields.begin(), sfields.end());
	return teq::Shape(slist);
}

inline bool equals (double scalar, const teq::iLeaf* leaf)
{
	return teq::IMMUTABLE == leaf->get_usage() &&
		leaf->to_string() == fmts::to_string(scalar);
}

inline bool equals (const Variable& var, const teq::iLeaf* leaf)
{
	return (Variable::kLabel != var.nullable_label_case() || var.label() == leaf->to_string()) &&
		(Variable::kDtype != var.nullable_dtype_case() || var.dtype() == leaf->type_label()) &&
		(0 == var.shape_size() || to_shape(var.shape()).compatible_after(leaf->shape(), 0));
}

void json_parse (Node& condition, std::istream& json_in);

}

#endif // QUERY_PARSE_HPP
