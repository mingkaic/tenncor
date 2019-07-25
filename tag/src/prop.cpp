#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = typeid(PropTag).hash_code();

PropertyRegistry& get_property_reg (void)
{
	static PropertyRegistry registry;
	return registry;
}

}

#endif
