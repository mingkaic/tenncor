/// group.hpp
/// tag
///
/// Purpose:
/// Implement group tag
///

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "tag/tag.hpp"

#ifndef TAG_GROUP_HPP
#define TAG_GROUP_HPP

namespace tag
{

/// Tensor ref set
using TensSetT = std::unordered_set<TensKey,TensKeyHash>;

/// GroupTag define subgraphs/groups of nodes with a structural significance
/// Groups are ordered tags, subsequent group tags
/// (obtained through absorption) often denote supergraphs of prior groups
///		e.g.: given tensor X, tag X with 'sum', then tag X with 'prod',
///		results in X having a collective [GroupTag:['sum','prod']]
///		ordered tag retains information that 'sum' is a subgraph of 'prod'
struct GroupTag final : public iTag
{
	GroupTag (std::string init_label) : labels_({init_label}) {}

	/// Implementation of iTag
	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	/// Implementation of iTag
	void absorb (TagptrT&& other) override
	{
		std::set<std::string>& olabels =
			static_cast<GroupTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
	}

	/// Implementation of iTag
	TagRepsT get_tags (void) const override;

private:
	std::set<std::string> labels_;

	static size_t tag_id_;
};

/// TagRegistry wrapper to tag tensors groups and
/// store reverse group-tensor association
struct GroupRegistry final
{
	GroupRegistry (TagRegistry& registry = get_reg()) : tag_reg_(registry) {}

	/// Bidirectionally assocate tensor and group
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

	/// Map group label to tensors
	std::unordered_map<std::string,TensSetT> groups_;

	/// Internal tag registry used to retrieve tensor-group association
	TagRegistry& tag_reg_;
};

/// Return reference to global group registry
GroupRegistry& get_group_reg (void);

/// Identifier of groups tag
const std::string groups_key = get_reg().register_tagr("groups",
[](teq::TensrefT ref, std::string tag)
{
	get_group_reg().group_tag(ref, tag);
});

/// Recursive add every nodes under root's graph to specified group
/// ignoring subgraphs of roots in stops set
void recursive_group_tag (teq::TensptrT tens, std::string group,
	teq::TensSetT stops,
	GroupRegistry& registry = get_group_reg());

/// Map group label to unique ids specifying different
/// adjacent groups of the same label
using AGroupsT = std::map<std::string,std::unordered_set<std::string>>;

/// Map tensors to groups
using AdjMapT = std::unordered_map<teq::iTensor*,AGroupsT>;

/// Populate out with every grouped-node under roots subgraphs
/// and associate to unique adjacent groups
void adjacencies (AdjMapT& out, teq::TensptrsT roots,
	GroupRegistry& registry = get_group_reg());

/// Subgraph group encapsulation
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

	/// Group label
	std::string group_;

	/// Group content
	teq::TensSetT content_;

	// todo: order subgraphs children somehow
	/// Children of the group
	std::unordered_map<teq::iTensor*,teq::TensptrT> children_;
};

/// Smart pointer of the subgraph
using SgraphptrT = std::shared_ptr<Subgraph>;

/// Set of subgraphs
using SubgraphsT = std::unordered_set<SgraphptrT>;

/// Root of the subgraph associated with the subgraph representations
using SubgraphAssocsT = std::unordered_map<teq::iTensor*,SubgraphsT>;

/// Populate subgraph associations using adjacency output
/// out associates every tensor under an adjacent group
/// to its subgraph representations
void beautify_groups (SubgraphAssocsT& out, const AdjMapT& adjs);

/// Transform subgraph content tensor-subgraph representation associations
/// to subgraph root-subgraph representation associations
void filter_head (SubgraphAssocsT& out, const SubgraphAssocsT& assocs);

}

#endif // TAG_GROUP_HPP
