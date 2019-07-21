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

void adjacencies (AdjMapT& out, ade::TensT roots,
	GroupRegistry& registry)
{
	ade::HeightMatrix mat(roots);

	boost::uuids::random_generator uuid_gen;
	for (auto it = mat.funcs_.rbegin(), et = mat.funcs_.rend();
		it != et; ++it)
	{
		auto& funcs = *it;
		for (ade::iFunctor* func : funcs)
		{
			TagRepsT tags = registry.tag_reg_.get_tags(func);
			std::vector<std::string> groups;
			if (estd::get(groups, tags, groups_key))
			{
				auto& children = func->get_children();
				std::unordered_set<ade::iTensor*> uchildren;
				std::transform(children.begin(), children.end(),
					std::inserter(uchildren, uchildren.end()),
					[](const ade::FuncArg& arg)
					{
						return arg.get_tensor().get();
					});

				auto& mygroups = out[func];
				for (std::string group : groups)
				{
					// set or inherit from parent, the unique gid of func
					std::unordered_set<std::string> gids;
					// try to inherit unique gid
					if (false == estd::get(gids, mygroups, group))
					{
						gids = {boost::uuids::to_string(uuid_gen())};
						mygroups.emplace(group, gids);
					}

					auto& same_group = registry.groups_[group];
					for (ade::iTensor* child : uchildren)
					{
						// propagate unique gid set to child of same group
						auto it = same_group.find(TensKey(child));
						if (same_group.end() != it && false == it->expired())
						{
							out[child][group].insert(gids.begin(), gids.end());
						}
					}
				}
			}
		}
	}

	for (ade::iLeaf* leaf : mat.leaves_)
	{
		auto tags = registry.tag_reg_.get_tags(leaf);
		std::vector<std::string> groups;
		if (estd::get(groups, tags, groups_key))
		{
			auto& mygroups = out[leaf];
			for (std::string group : groups)
			{
				// set unique gids if there are no inherited groups
				if (false == estd::has(mygroups, group))
				{
					mygroups.emplace(group,
						std::unordered_set<std::string>{
							boost::uuids::to_string(uuid_gen()),
						});
				}
			}
		}
	}
}

void beautify_groups (SubgraphAssocsT& out, const AdjMapT& adjs)
{
	std::unordered_map<std::string,SgraphptrT> sgraphs;
	for (auto& gpair : adjs)
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
