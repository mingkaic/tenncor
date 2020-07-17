//
/// parse.hpp
/// query
///
/// Purpose:
/// Define subgraph filtering condition parsing
///

#ifndef QUERY_PARSE_HPP
#define QUERY_PARSE_HPP

#include "teq/teq.hpp"

#include "query/query.pb.h"

namespace query
{

void json_parse (Node& condition, std::istream& json_in);

}

#endif // QUERY_PARSE_HPP
