#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

TagRegistry& get_reg (void)
{
	static TagRegistry registry;
	return registry;
}

}

#endif
