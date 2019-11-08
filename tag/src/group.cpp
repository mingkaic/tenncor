#include "tag/group.hpp"

#ifdef TAG_GROUP_HPP

namespace tag
{

size_t GroupTag::tag_id_ = typeid(GroupTag).hash_code();

TagRepsT GroupTag::get_tags (void) const
{
	TagRepsT out;
	out.emplace(groups_key,
		std::vector<std::string>(labels_.begin(), labels_.end()));
	return out;
}

GroupRegistry& get_group_reg (void)
{
	static GroupRegistry registry;
	return registry;
}

void recursive_group_tag (teq::TensptrT tens, std::string group,
	teq::TensSetT stops, GroupRegistry& registry)
{
	recursive_tag(tens, stops,
		[&](teq::TensrefT ref)
		{
			registry.group_tag(ref, group);
		});
}

void adjacencies (AdjMapT& out, teq::TensptrsT roots,
	GroupRegistry& registry)
{
	teq::HeightMatrix mat(roots);

	boost::uuids::random_generator uuid_gen;
	for (auto it = mat.funcs_.rbegin(), et = mat.funcs_.rend();
		it != et; ++it)
	{
		auto& funcs = *it;
		for (teq::iFunctor* func : funcs)
		{
			TagRepsT tags = registry.tag_reg_.get_tags(func);
			std::vector<std::string> groups;
			if (estd::get(groups, tags, groups_key))
			{
				auto children = func->get_children();
				teq::TensSetT uchildren;
				std::transform(children.begin(), children.end(),
					std::inserter(uchildren, uchildren.end()),
					[](const teq::iEdge& arg)
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
					for (teq::iTensor* child : uchildren)
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

	for (teq::iLeaf* leaf : mat.leaves_)
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
		teq::iTensor* tens = gpair.first;
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
		for (teq::iTensor* content : sg.second->content_)
		{
			out[content].emplace(sg.second);
		}
	}
}

void filter_head (SubgraphAssocsT& out, const SubgraphAssocsT& assocs)
{
	teq::GraphStat stat;
	for (auto& assoc_pair : assocs)
	{
		assoc_pair.first->accept(stat);
	}
	std::unordered_map<tag::SgraphptrT,teq::iTensor*> revhead;
	for (auto& sgpair : assocs)
	{
		const SubgraphsT& subgraphs = sgpair.second;
		for (const SgraphptrT& subgraph : subgraphs)
		{
			if (estd::has(revhead, subgraph))
			{
				teq::iTensor*& oldhead = revhead[subgraph];
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
