#include <optional>
#include <sstream>

#include <boost/property_tree/json_parser.hpp>

#include "subgraph_match/transform.hpp"

#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

namespace opt
{

const std::regex comment_transform("([^\\\\]?//.*)");

std::string strip_comment (std::string line)
{
    std::smatch sm;
    if (std::regex_search(line, sm, comment_transform))
    {
        size_t comment_begin = line.size() - sm[0].length();
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
    Transform transform;
    std::string transform_half = line.substr(0, idelim);
    std::string result_half = line.substr(idelim + tuple_delim.size());
    transform.pheight_ = 1;
    size_t i = 0;
    while ((i = transform_half.find(line_delim, i)) != std::string::npos)
    {
        ++transform.pheight_;
        i += line_delim.size();
    }
    fmts::trim(transform_half);
    fmts::trim(result_half);
    transform.pattern_ = std::regex(transform_half);
    transform.simplification_ = result_half;
    out = transform;
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

#endif // OPT_PARSE_HPP
