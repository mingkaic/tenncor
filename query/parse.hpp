#ifndef QUERY_PARSE_HPP
#define QUERY_PARSE_HPP

#include "query/query.pb.h"

namespace query
{

void json_parse (Node& condition, std::istream& json_in);

}

#endif // QUERY_PARSE_HPP
