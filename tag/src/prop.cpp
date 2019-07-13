#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = TagCollective::register_tag<PropTag>();

void property_tag (ade::TensrefT tens, std::string property)
{
	get_reg().get_collective(tens).add(std::make_unique<PropTag>(property));
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
