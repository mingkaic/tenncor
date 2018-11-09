#include "adhoc/llo/api.hpp"

struct iLayer
{
	iLayer (std::string label) : label_(label) {}

	virtual ~iLayer (void) {}

	virtual std::vector<llo::DataNode> get_variables (void) const = 0;

	std::string label_;
};
