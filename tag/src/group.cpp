#include "tag/group.hpp"

#ifdef TAG_GROUP_HPP

namespace tag
{

using RefMapT = std::unordered_map<ade::iTensor*,ade::TensrefT>;

struct Grouper final : public ade::iTraveler
{
	Grouper (std::string group,
		std::unordered_set<ade::iTensor*> stops) :
		group_(group), stops_(stops.begin(), stops.end()) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (stops_.end() == stops_.find(leaf))
		{
			auto it = owners_.find(leaf);
			if (owners_.end() == it)
			{
				logs::fatal("failed to get reference to leaf in group traveler");
			}
			tag::group_tag(it->second, group_);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (stops_.end() == stops_.find(func))
		{
			auto it = owners_.find(func);
			if (owners_.end() == it)
			{
				logs::fatal("failed to get reference to leaf in group traveler");
			}
			auto& children = func->get_children();
			for (auto& child : children)
			{
				ade::TensptrT tens = child.get_tensor();
				owners_.emplace(tens.get(), tens);
				tens->accept(*this);
			}
			tag::group_tag(it->second, group_);
		}
	}

	RefMapT owners_;

	std::string group_;

	std::unordered_set<ade::iTensor*> stops_;
};

void recursive_group_tag (ade::TensrefT tens, std::string group,
	std::unordered_set<ade::iTensor*> stops)
{
	if (tens.expired())
	{
		logs::fatal("cannot recursive group tag with expired tensor ref");
	}
	Grouper trav(group, stops);
	auto tensor = tens.lock().get();
	trav.owners_.emplace(tensor, tens);
	tensor->accept(trav);
}

}

#endif
