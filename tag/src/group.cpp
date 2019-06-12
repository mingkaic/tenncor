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

boost::uuids::random_generator AdjacentGroups::uuid_gen_;

void beautify_groups (SubgraphsT& out, const AdjacentGroups& adjgroups)
{
	std::unordered_map<std::string,SgraphptrT> sgraphs;
	for (auto& gpair : adjgroups.adjs_)
	{
		ade::iTensor* tens = gpair.first;
		for (auto& idpair : gpair.second)
		{
			std::string group = idpair.first;
			for (const std::string& gid : idpair.second)
			{
				auto it = sgraphs.find(gid);
				if (sgraphs.end() == it)
				{
					sgraphs.emplace(gid, std::make_shared<Subgraph>(group));
				}
				tens->accept(*sgraphs[gid]);
			}
		}
	}

	for (auto& sg : sgraphs)
	{
		for (ade::iTensor* content : sg.second->content_)
		{
			out.emplace(content, sg.second);
		}
	}
}

}

#endif
