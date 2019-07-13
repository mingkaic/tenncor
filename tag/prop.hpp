#include "tag/tag.hpp"

#ifndef TAG_PROP_HPP
#define TAG_PROP_HPP

namespace tag
{

const std::string props_key = "properties";

// some property tags
const std::string commutative_tag = "commutative";

const std::string immutable_tag = "immutable";

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

struct PropertyRegistry final
{
	PropertyRegistry (TagRegistry& registry = get_reg()) :
		tag_reg_(registry) {}

	void property_tag (ade::TensrefT tens, std::string property)
	{
		tag_reg_.add_tag(tens, TagptrT(new PropTag(property)));
	}

	bool has_property (const ade::iTensor* tens, std::string property)
	{
		auto reps = tag_reg_.get_tags(tens);
		auto it = reps.find(props_key);
		if (reps.end() == it)
		{
			return false;
		}
		return estd::arr_has(it->second, property);
	}

	TagRegistry& tag_reg_;
};

PropertyRegistry& get_property_reg (void);

}

#endif // TAG_PROP_HPP
