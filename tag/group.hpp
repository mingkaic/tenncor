#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "tag/tag.hpp"

#ifndef TAG_GROUP_HPP
#define TAG_GROUP_HPP

namespace tag
{

using TensSetT = std::unordered_set<TensKey,TensKeyHash>;

/// GroupTag define subgraphs/groups of nodes with a structural significance
/// Groups are ordered tags, subsequent group tags
/// (obtained through absorption) often denote supergraphs of prior groups
///		e.g.: given tensor X, tag X with 'sum', then tag X with 'mlp',
///		results in X having a collective [GroupTag:['sum','mlp']]
///		ordered tag retains information that 'sum' is a subgraph of 'mlp'
struct GroupTag final : public iTag
{
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

	TagRepsT get_tags (void) const override;

private:
	std::set<std::string> labels_;

	static size_t tag_id_;
};

struct GroupRegistry final
{
	GroupRegistry (TagRegistry& registry = get_reg()) : tag_reg_(registry) {}

	void group_tag (teq::TensrefT tens, std::string tag)
	{
		tag_reg_.add_tag(tens, TagptrT(new GroupTag(tag)));

		auto& gtens = groups_[tag];
		auto it = gtens.find(TensKey(tens.lock().get()));
		// clear out previous entry that is expired
		if (gtens.end() != it && it->expired())
		{
			gtens.erase(tens.lock().get());
		}
		gtens.emplace(tens);
	}

	std::unordered_map<std::string,TensSetT> groups_;

	TagRegistry& tag_reg_;
};

GroupRegistry& get_group_reg (void);

const std::string groups_key = get_reg().register_tagr("groups",
[](teq::TensrefT ref, std::string tag)
{
	get_group_reg().group_tag(ref, tag);
});

void recursive_group_tag (teq::TensptrT tens, std::string group,
	std::unordered_set<teq::iTensor*> stops,
	GroupRegistry& registry = get_group_reg());

using AGroupsT = std::map<std::string,std::unordered_set<std::string>>;

using AdjMapT = std::unordered_map<teq::iTensor*,AGroupsT>;

void adjacencies (AdjMapT& out, teq::TensT roots,
	GroupRegistry& registry = get_group_reg());

struct Subgraph final : public teq::iTraveler
{
	Subgraph (std::string group) : group_(group) {}

	/// Implementation of iTraveler
	void visit (teq::iLeaf* leaf) override
	{
		if (false == estd::has(content_, leaf))
		{
			content_.emplace(leaf);
			children_.erase(leaf);
		}
	}

	/// Implementation of iTraveler
	void visit (teq::iFunctor* func) override
	{
		if (false == estd::has(content_, func))
		{
			content_.emplace(func);
			children_.erase(func);

			auto& children = func->get_children();
			for (auto& child : children)
			{
				auto tens = child.get_tensor();
				if (false == estd::has(content_, tens.get()))
				{
					children_.emplace(tens.get(), tens);
				}
			}
		}
	}

	std::string group_;

	std::unordered_set<teq::iTensor*> content_;

	// todo: order subgraphs children somehow
	std::unordered_map<teq::iTensor*,teq::TensptrT> children_;
};

using SgraphptrT = std::shared_ptr<Subgraph>;

using SubgraphsT = std::unordered_set<SgraphptrT>;

using SubgraphAssocsT = std::unordered_map<teq::iTensor*,SubgraphsT>;

void beautify_groups (SubgraphAssocsT& out, const AdjMapT& adjs);

// look for associations where the tensor key is the
// max height tensor of the mapped subgraph
// dump filtered associations in out
void filter_head (SubgraphAssocsT& out, const SubgraphAssocsT& assocs);

}

#endif // TAG_GROUP_HPP
