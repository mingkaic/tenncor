#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = TagCollective::register_tag<PropTag>();

void property_tag (ade::TensrefT tens, std::string property)
{
	if (tens.expired())
	{
		logs::fatal("cannot property tag with expired tensor ref");
	}
	Registry::registry[tens].add(std::make_unique<tag::PropTag>(property));
}

}

#endif
