#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

std::unordered_set<size_t> TagCollective::tag_types_;

size_t GroupTag::tag_id_ = TagCollective::register_tag<GroupTag>();

std::unordered_map<std::string,TensSetT> GroupTag::groups_;

static std::unordered_map<ade::iTensor*,TagCollective> registry;

void group_tag (ade::iTensor* tens, std::string group)
{
	GroupTag::groups_[group].emplace(tens);
	registry[tens].add(std::make_unique<tag::GroupTag>(group));
}

}

#endif
