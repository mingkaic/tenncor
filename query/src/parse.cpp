#include <google/protobuf/util/json_util.h>

#include "logs/logs.hpp"

#include "query/parse.hpp"

#ifdef QUERY_PARSE_HPP

namespace query
{

void json_parse (Node& condition, std::istream& json_in)
{
	std::string jstr(std::istreambuf_iterator<char>(json_in), {});
	google::protobuf::util::JsonParseOptions options;
	options.ignore_unknown_fields = true;
	if (google::protobuf::util::Status::OK !=
		google::protobuf::util::JsonStringToMessage(
			jstr, &condition, options))
	{
		logs::fatal("failed to parse json condition");
	}
}

}

#endif
