#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "tag/tag.hpp"

#ifndef TAG_GROUP_HPP
#define TAG_GROUP_HPP

namespace tag
{

using TensSetT = std::unordered_set<TensKey,TensKeyHash>;

const std::string groups_key = "groups";

/// GroupTag define subgraphs/groups of nodes with a structural significance
/// Groups are ordered tags, subsequent group tags
/// (obtained through absorption) often denote supergraphs of prior groups
///		e.g.: given tensor X, tag X with 'sum', then tag X with 'mlp',
///		results in X having a collective [GroupTag:['sum','mlp']]
///		ordered tag retains information that 'sum' is a subgraph of 'mlp'
struct GroupTag final : public iTag
{
	static std::unordered_map<std::string,TensSetT> groups_;

	GroupTag (std::string init_label) : labels_({init_label}) {}

	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	void absorb (TagptrT&& other) override
	{
		std::set<std::string>& olabels =
			static_cast<GroupTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
	}

	TagRepsT get_tags (void) const override
	{
		TagRepsT out;
		out.emplace(groups_key,
			std::vector<std::string>(labels_.begin(), labels_.end()));
		return out;
	}

private:
	std::set<std::string> labels_;

	static size_t tag_id_;
};

void group_tag (ade::TensrefT tens, std::string group);

void recursive_group_tag (ade::TensrefT tens, std::string group,
	std::unordered_set<ade::iTensor*> stops);

using AGroupsT = std::map<std::string,std::unordered_set<std::string>>;

struct AdjacentGroups final : public ade::iTraveler
{
	static boost::uuids::random_generator uuid_gen_;

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == util::has(visited_, leaf))
		{
			visited_.emplace(leaf);
			auto tags = get_tags(leaf);
			std::vector<std::string> groups;
			if (util::get(groups, tags, groups_key))
			{
				auto& mygroups = adjs_[leaf];
				for (std::string group : groups)
				{
					// set unique gids if there are no inherited groups
					if (false == util::has(mygroups, group))
					{
						mygroups.emplace(group,
							std::unordered_set<std::string>{
								boost::uuids::to_string(uuid_gen_()),
							});
					}
				}
			}
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == util::has(visited_, func))
		{
			visited_.emplace(func);
			auto& children = func->get_children();
			std::unordered_set<ade::iTensor*> uchildren;
			std::transform(children.begin(), children.end(),
				std::inserter(uchildren, uchildren.end()),
				[](const ade::FuncArg& arg)
				{
					return arg.get_tensor().get();
				});
			TagRepsT tags = get_tags(func);
			std::vector<std::string> groups;
			if (util::get(groups, tags, groups_key))
			{
				auto& mygroups = adjs_[func];
				for (std::string group : groups)
				{
					// set or inherit from parent, the unique gid of func
					std::unordered_set<std::string> gids;
					// try to inherit unique gid
					if (false == util::get(gids, mygroups, group))
					{
						gids = {boost::uuids::to_string(uuid_gen_())};
						mygroups.emplace(group, gids);
					}

					auto& same_group = GroupTag::groups_[group];
					for (ade::iTensor* child : uchildren)
					{
						// propagate unique gid set to child of same group
						auto it = same_group.find(TensKey(child));
						if (same_group.end() != it && false == it->expired())
						{
							adjs_[child][group].insert(gids.begin(), gids.end());
						}
					}
				}
			}
			// process children adjacency
			for (ade::iTensor* child : uchildren)
			{
				child->accept(*this);
			}
		}
	}

	std::unordered_set<ade::iTensor*> visited_;

	std::unordered_map<ade::iTensor*,AGroupsT> adjs_;
};

struct Subgraph final : public ade::iTraveler
{
	Subgraph (std::string group) : group_(group) {}

	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (false == util::has(content_, leaf))
		{
			content_.emplace(leaf);
			children_.erase(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (false == util::has(content_, func))
		{
			content_.emplace(func);
			children_.erase(func);

			auto& children = func->get_children();
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				if (false == util::has(content_, tens.get()))
				{
					children_.emplace(tens.get(), tens);
				}
			}
		}
	}

	std::string group_;

	std::unordered_set<ade::iTensor*> content_;

	// todo: order subgraphs children somehow
	std::unordered_map<ade::iTensor*,ade::TensptrT> children_;
};

using SgraphptrT = std::shared_ptr<Subgraph>;

using SubgraphsT = std::unordered_map<ade::iTensor*,SgraphptrT>;

void beautify_groups (SubgraphsT& out, const AdjacentGroups& adjgroups);

}

#endif // TAG_GROUP_HPP
