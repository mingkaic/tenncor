#include "ead/matcher/parse.hpp"

#ifdef OPT_PARSE_HPP

namespace opt
{

static const std::regex define_pattern("#define (\\w+) (.*)");

static void replace_all (std::string& s,
	std::string target, std::string repl)
{
	size_t i = 0;
	while ((i = s.find(target, i)) != std::string::npos)
	{
		s.replace(i, target.size(), repl);
		i += repl.size();
	}
}

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

std::vector<std::string> preprocess (std::istream& in)
{
	std::vector<std::string> out;
	std::unordered_map<std::string,std::string> defined_map;
	std::string line;
	while (std::getline(in, line))
	{
		line = strip_comment(line);
		std::smatch sm;
		if (std::regex_match(line, sm, define_pattern))
		{
			assert(sm.size() == 3);
			defined_map.emplace(sm[1].str(), sm[2].str());
		}
		else if (line.size() > 0)
		{
			for (auto& defpair : defined_map)
			{
				replace_all(line, defpair.first, defpair.second);
			}
			out.push_back(line);
		}
	}
	return out;
}

Transform parse (std::string line)
{
	size_t idelim = line.find(tuple_delim);
	if (idelim == std::string::npos)
	{
		logs::fatalf("cannot parse non-statement %s", line.c_str());
	}
	return Transform(line);
}

TransformsT parse_stream (std::istream& in)
{
	std::vector<std::string> lines = preprocess(in);
	TransformsT out;
	for (std::string line : lines)
	{
		out.push_back(parse(line));
	}
	return out;
}

}

#endif
