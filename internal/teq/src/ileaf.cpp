#include <unordered_map>

#include "estd/estd.hpp"

#include "internal/teq/ileaf.hpp"

#ifdef TEQ_ILEAF_HPP

namespace teq
{

static const types::StrUMapT<Usage> named_usages = {
	{"constant", IMMUTABLE},
	{"variable", VARUSAGE},
	{"placeholder", PLACEHOLDER},
};

static const std::unordered_map<Usage,std::string,estd::EnumHash> usage_names = {
	{IMMUTABLE, "constant"},
	{VARUSAGE, "variable"},
	{PLACEHOLDER, "placeholder"},
};

Usage get_named_usage (const std::string& name)
{
	return estd::try_get(named_usages, name, UNKNOWN_USAGE);
}

std::string get_usage_name (Usage usage)
{
	return estd::try_get(usage_names, usage, "");
}

}

#endif
