#include <unordered_map>

#include "estd/estd.hpp"

#include "teq/ileaf.hpp"

#ifdef TEQ_ILEAF_HPP

namespace teq
{

static const std::unordered_map<std::string,Usage> named_usages = {
	{"constant", Immutable},
	{"variable", Variable},
	{"placeholder", Placeholder},
};

static const std::unordered_map<Usage,std::string> usage_names = {
	{Immutable, "constant"},
	{Variable, "variable"},
	{Placeholder, "placeholder"},
};

Usage get_named_usage (std::string name)
{
	return estd::try_get(named_usages, name, Unknown);
}

std::string get_usage_name (Usage usage)
{
	return estd::try_get(usage_names, usage, "");
}

}

#endif
