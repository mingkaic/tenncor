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

	void absorb (std::unique_ptr<iTag>&& other) override
	{
		std::set<std::string>& olabels =
			static_cast<GroupTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
		other.release();
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

std::unordered_set<ade::iTensor*> adjacent_group (
	ade::iTensor* tens, std::string group);

}

#endif // TAG_GROUP_HPP
