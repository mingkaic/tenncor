#include "tag/group.hpp"

#ifdef TAG_GROUP_HPP

namespace tag
{

using RefMapT = std::unordered_map<ade::iTensor*,ade::TensrefT>;

size_t GroupTag::tag_id_ = TagCollective::register_tag<GroupTag>();

std::unordered_map<std::string,TensSetT> GroupTag::groups_;

void group_tag (ade::TensrefT tens, std::string group)
{
	if (tens.expired())
	{
		logs::fatal("cannot group tag with expired tensor ref");
	}
	GroupTag::groups_[group].emplace(tens);
	Registry::registry[tens].add(std::make_unique<tag::GroupTag>(group));
}

struct Grouper final : public ade::iTraveler
{
	Grouper (std::string group,
		std::unordered_set<ade::iTensor*> stops) :
		group_(group), stops_(stops.begin(), stops.end()) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == util::has(stops_, leaf))
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
		if (false == util::has(stops_, func))
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

struct Adjacents final : public ade::iTraveler
{
	Adjacents (std::string group) : group_(group) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == util::has(out_, leaf))
		{
			auto tags = Registry::registry[func].get_tags();
			auto it = tags.find(groups_key);
			if (tags.end() == it)
			{
				return;
			}
			if (it->second.end() == std::find(
				it->second.begin(), it->second.end(),
				[&](const std::string& k) { return k == group; }))
			{
				return;
			}
			out_.emplace(func);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == util::has(out_, func))
		{
			auto tags = Registry::registry[func].get_tags();
			auto it = tags.find(groups_key);
			if (tags.end() == it)
			{
				return;
			}
			if (it->second.end() == std::find(
				it->second.begin(), it->second.end(),
				[&](const std::string& k) { return k == group; }))
			{
				return;
			}
			auto& children = func->get_children();
			for (auto& child : children)
			{
				child.get_tensor()->accept(*this);
			}
			out_.emplace(func);
		}
	}

	std::string group_;

	std::unordered_set<ade::iTensor*> out_;
};

std::unordered_set<ade::iTensor*> adjacent_group (
	ade::iTensor* tens, std::string group)
{
	Adjacents adj(group);
	tens->accept(adj);
	return adj.out_;
}

}

#endif
