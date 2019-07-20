#include "tag/group.hpp"

#ifdef TAG_GROUP_HPP

namespace tag
{

using RefMapT = std::unordered_map<ade::iTensor*,ade::TensrefT>;

size_t GroupTag::tag_id_ = typeid(GroupTag).hash_code();

GroupRegistry& get_group_reg (void)
{
	static GroupRegistry registry;
	return registry;
}

struct Grouper final : public ade::iTraveler
{
	Grouper (std::string group, std::unordered_set<ade::iTensor*> stops,
		GroupRegistry& registry) :
		group_(group),
		stops_(stops.begin(), stops.end()),
		registry_(registry) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == estd::has(stops_, leaf))
		{
			auto it = owners_.find(leaf);
			if (owners_.end() == it)
			{
				logs::fatal("failed to get reference to leaf in group traveler");
			}
			registry_.group_tag(it->second, group_);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == estd::has(stops_, func))
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
			registry_.group_tag(it->second, group_);
		}
	}

	RefMapT owners_;

	std::string group_;

	std::unordered_set<ade::iTensor*> stops_;

	GroupRegistry& registry_;
};

void recursive_group_tag (ade::TensrefT tens, std::string group,
	std::unordered_set<ade::iTensor*> stops, GroupRegistry& registry)
{
	if (tens.expired())
	{
		logs::fatal("cannot recursive group tag with expired tensor ref");
	}
	Grouper trav(group, stops, registry);
	auto tensor = tens.lock().get();
	trav.owners_.emplace(tensor, tens);
	tensor->accept(trav);
}

boost::uuids::random_generator AdjacentGroups::uuid_gen_;

void beautify_groups (SubgraphAssocsT& out, const AdjacentGroups& adjgroups)
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
				sgraphs.try_emplace(gid,
					std::make_shared<Subgraph>(group));
				tens->accept(*sgraphs[gid]);
			}
		}
	}

	for (auto& sg : sgraphs)
	{
		for (ade::iTensor* content : sg.second->content_)
		{
			out[content].emplace(sg.second);
		}
	}
}

void filter_head (SubgraphAssocsT& out, const SubgraphAssocsT& assocs)
{
	ade::GraphStat stat;
	for (auto& assoc_pair : assocs)
	{
		assoc_pair.first->accept(stat);
	}
	std::unordered_map<tag::SgraphptrT,ade::iTensor*> revhead;
	for (auto& sgpair : assocs)
	{
		const SubgraphsT& subgraphs = sgpair.second;
		for (const SgraphptrT& subgraph : subgraphs)
		{
			if (estd::has(revhead, subgraph))
			{
				ade::iTensor*& oldhead = revhead[subgraph];
				if (stat.graphsize_[sgpair.first].upper_ >
					stat.graphsize_[oldhead].upper_)
				{
					// add sgpair.first as head if it has greater maxheight
					revhead[subgraph] = sgpair.first;
				}
			}
			else
			{
				revhead.emplace(subgraph, sgpair.first);
			}
		}
	}
	out.clear();
	for (auto& revpair : revhead)
	{
		out[revpair.second].emplace(revpair.first);
	}
}

}

#endif
