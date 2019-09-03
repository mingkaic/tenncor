#include "tag/prop.hpp"

#ifdef TAG_PROP_HPP

namespace tag
{

size_t PropTag::tag_id_ = typeid(PropTag).hash_code();

TagRepsT PropTag::get_tags (void) const
{
	TagRepsT out;
	out.emplace(props_key, std::vector<std::string>(
		labels_.begin(), labels_.end()));
	return out;
}

bool PropertyRegistry::has_property (const ade::iTensor* tens, std::string property) const
{
	auto reps = tag_reg_.get_tags(tens);
	auto it = reps.find(props_key);
	if (reps.end() == it)
	{
		return false;
	}
	return estd::arr_has(it->second, property);
}

PropertyRegistry& get_property_reg (void)
{
	static PropertyRegistry registry;
	return registry;
}

}

#endif
