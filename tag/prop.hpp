#include "tag/tag.hpp"

#ifndef TAG_PROP_HPP
#define TAG_PROP_HPP

namespace tag
{

const std::string props_key = "properties";

/// PropTag (properties tag) define node properties
struct PropTag final : public iTag
{
	PropTag (std::string init_label) : labels_({init_label}) {}

	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	void absorb (TagptrT&& other) override
	{
		std::unordered_set<std::string>& olabels =
			static_cast<PropTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
	}

	TagRepsT get_tags (void) const override
	{
		TagRepsT out;
		out.emplace(props_key, std::vector<std::string>(
			labels_.begin(), labels_.end()));
		return out;
	}

private:
	std::unordered_set<std::string> labels_;

	static size_t tag_id_;
};

void property_tag (ade::TensrefT tens, std::string property);

bool has_property (const ade::iTensor* tens, std::string property);

// some property tags
const std::string commutative_tag = "commutative";

const std::string immutable_tag = "immutable";

}

#endif // TAG_PROP_HPP
