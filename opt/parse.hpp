///
/// parse.hpp
/// opt
///
/// Purpose:
/// Define interfaces to build extensions of TEQ graphs
/// and wrap around C parser
///

#include "opt/optimize.hpp"

#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

namespace opt
{

using BuildTargetF = std::function<TargptrT(::TreeNode*)>;

/// Return all parsed optimization rules of string content
CversionCtx parse (std::string content, BuildTargetF parse_target);

/// Return all parsed optimization rules of a file
CversionCtx parse_file (std::string filename, BuildTargetF parse_target);

}

#endif // OPT_PARSE_HPP
