#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

std::unordered_set<size_t> TagCollective::tag_types_;

std::unordered_map<TensKey,TagCollective,
	TensKeyHash> Registry::registry; // todo: make thread-safe

TagRepsT get_tags (ade::iTensor* tens)
{
	auto it = Registry::registry.find(TensKey(tens));
	if (Registry::registry.end() == it || it->first.expired())
	{
		return {};
	}
	return it->second.get_tags();
}

}

#endif
