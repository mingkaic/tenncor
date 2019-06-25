#include "tag/tag.hpp"

#ifdef TAG_TAG_HPP

namespace tag
{

std::unordered_map<TensKey,TagCollective,
	TensKeyHash> Registry::registry; // todo: make thread-safe

TagRepsT get_tags (const ade::iTensor* tens)
{
	auto it = Registry::registry.find(TensKey(tens));
	if (Registry::registry.end() == it || it->first.expired())
	{
		return {};
	}
	return it->second.get_tags();
}

void add_tag (ade::TensrefT tens, iTag* tag)
{
	if (tens.expired())
	{
		logs::fatal("cannot tag with expired tensor ref");
	}
	auto it = Registry::registry.find(TensKey(tens));
	// clear out previous entry that is expired
	if (Registry::registry.end() != it && it->first.expired())
	{
		Registry::registry.erase(tens.lock().get());
	}
	Registry::registry[tens].add(std::unique_ptr<iTag>(tag));
}

void erase (const ade::iTensor* tens)
{
	Registry::registry.erase(TensKey(tens));
}

void move_tags (const ade::iTensor* dest, const ade::iTensor* source)
{
	auto src_it = Registry::registry.find(TensKey(source));
	auto dest_it = Registry::registry.find(TensKey(dest));
	if (Registry::registry.end() == src_it || src_it->first.expired() ||
		Registry::registry.end() == dest_it || dest_it->first.expired())
	{
		return;
	}

	dest_it->second.absorb(std::move(src_it->second));
}

}

#endif
