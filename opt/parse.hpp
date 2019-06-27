#include "opt/optimize.hpp"

#ifndef OPT_PARSE_HPP
#define OPT_PARSE_HPP

namespace opt
{

struct RulesContext final
{
	// maps declared symbol to any id
	std::unordered_set<std::string> symbols_;

	// maps functor/group label to properties set
	std::unordered_map<std::string,
		std::unordered_set<std::string>> properties_;
};

struct iConverterBuilder
{
	virtual ~iConverterBuilder (void) = default;

	// return constant converter
	virtual CstConvertF build_cconv (void) const = 0;

	virtual ConvptrT build (const ::Subgraph* sg, const RulesContext& ctx) const = 0;

	// extended interface to create shaper
	virtual ade::CoordptrT shaperize (::NumList* list) const = 0;

	// extended interface to create coorder
	virtual ade::CoordptrT coorderize (::NumList* list) const = 0;
};

OptCtx parse (std::string content, const iConverterBuilder& builder);

OptCtx parse_file (std::string filename, const iConverterBuilder& builder);

}

#endif // OPT_PARSE_HPP
