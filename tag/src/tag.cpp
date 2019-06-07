#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

std::unordered_set<size_t> TagCollective::tag_types_;

size_t GroupTag::tag_id_ = TagCollective::register_tag<GroupTag>();

std::unordered_map<std::string,TensSetT> GroupTag::groups_;

static std::unordered_map<TensKey,TagCollective,TensKeyHash> registry; // todo: make thread-safe

TagRepsT get_tags (ade::iTensor* tens)
{
	auto it = registry.find(TensKey(tens));
	if (registry.end() == it || it->first.expired())
	{
		return {};
	}
	return it->second.get_tags();
}

void group_tag (ade::TensrefT tens, std::string group)
{
	if (tens.expired())
	{
		logs::fatal("cannot group tag with expired tensor ref");
	}
	GroupTag::groups_[group].emplace(tens);
	registry[tens].add(std::make_unique<tag::GroupTag>(group));
}

}

#endif
