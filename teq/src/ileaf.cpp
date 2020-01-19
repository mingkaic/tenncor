#include <unordered_map>

#include "estd/contain.hpp"

#include "teq/ileaf.hpp"

#ifdef TEQ_ILEAF_HPP

namespace teq
{

static const std::unordered_map<std::string,Usage> named_usages = {
	{"constant", IMMUTABLE},
	{"variable", VARUSAGE},
	{"placeholder", PLACEHOLDER},
};

static const std::unordered_map<Usage,std::string> usage_names = {
	{IMMUTABLE, "constant"},
	{VARUSAGE, "variable"},
	{PLACEHOLDER, "placeholder"},
};

Usage get_named_usage (std::string name)
{
	return estd::try_get(named_usages, name, UNKNOWN_USAGE);
}

std::string get_usage_name (Usage usage)
{
	return estd::try_get(usage_names, usage, "");
}

}

#endif
