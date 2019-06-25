#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = TagCollective::register_tag<PropTag>();

void property_tag (ade::TensrefT tens, std::string property)
{
	add_tag(tens, new PropTag(property));
}

bool has_property (const ade::iTensor* tens, std::string property)
{
	auto reps = get_tags(tens);
	auto it = reps.find(props_key);
	if (reps.end() == it)
	{
		return false;
	}
	return it->second.end() != std::find(
		it->second.begin(), it->second.end(), property);
}

}

#endif
