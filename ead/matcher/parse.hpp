#include "ead/matcher/transform.hpp"

#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

namespace opt
{

std::vector<std::string> preprocess (std::istream& in);

Transform parse (std::string line);

TransformsT parse_stream (std::istream& in);

}

#endif // OPT_PARSE_HPP
