#include "subgraph_match/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

static const std::regex comment_pattern("[^\\\\]?(//.*)");

static std::string strip_comment (std::string line)
{
	std::smatch sm;
	if (std::regex_search(line, sm, comment_pattern))
	{
		size_t comment_begin = line.size() - sm[1].length();
		line = line.substr(0, comment_begin);
	}
	return line;
}

std::optional<Transform> parse (std::string line)
{
	std::optional<Transform> out;
	line = strip_comment(line);
	if (0 == line.size())
	{
		return out;
	}
	size_t idelim = line.find(tuple_delim);
	if (idelim == std::string::npos)
	{
		return out;
	}
	out = Transform(line);
	return out;
}

TransformsT parse_lines (std::istream& in)
{
	TransformsT out;
	std::string line;
	while (std::getline(in, line))
	{
		auto transform = parse(line);
		if (transform)
		{
			out.push_back(*transform);
		}
	}
	return out;
}

}

#endif
