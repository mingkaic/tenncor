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

/// Global statements shared between all rule statements
struct RulesContext final
{
	/// Set of declared ANY typed rule node id
	std::unordered_set<std::string> symbols_;

	/// Map functor/group label to set of associated properties
	std::unordered_map<std::string,
		std::unordered_set<std::string>> properties_;
};

/// Interface for TEQ extensions to construct conversion rules
struct iConverterBuilder
{
	virtual ~iConverterBuilder (void) = default;

	/// Return converter func that identifies extended TEQ specific constants
	virtual CstConvertF build_cconv (void) const = 0;

	/// Return extended TEQ builders specific to a
	/// target rule graph and statement contexts
	virtual ConvptrT build (const ::Subgraph* sg, const RulesContext& ctx) const = 0;

	/// Return shape mapper given parsed C representation
	virtual teq::CvrtptrT shaperize (::NumList* list) const = 0;

	/// Return coordinate mapper given parsed C representation
	virtual teq::CvrtptrT coorderize (::NumList* list) const = 0;
};

/// Return all parsed optimization rules of string content
OptCtx parse (std::string content, const iConverterBuilder& builder);

/// Return all parsed optimization rules of a file
OptCtx parse_file (std::string filename, const iConverterBuilder& builder);

}

#endif // OPT_PARSE_HPP
