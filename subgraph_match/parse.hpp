#ifdef __cpp_lib_optional
#include <optional>
#else
#include <experimental/optional>
namespace std
{
template <typename T>
using optional = ::std::experimental::optional<T>;
}
#endif

#include "subgraph_match/transform.hpp"

#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

namespace opt
{

std::optional<Transform> parse (std::string line);

TransformsT parse_lines (std::istream& in);

}

#endif // OPT_PARSE_HPP
