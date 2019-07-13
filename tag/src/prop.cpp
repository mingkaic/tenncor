#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = get_reg().register_tag<PropTag>();

void property_tag (ade::TensrefT tens, std::string property)
{
	get_reg().add_tag(tens, std::make_unique<PropTag>(property));
}

bool has_property (const ade::iTensor* tens, std::string property)
{
	auto reps = get_reg().get_tags(tens);
	auto it = reps.find(props_key);
	if (reps.end() == it)
	{
		return false;
	}
	return estd::arr_has(it->second, property);
}

}

#endif
