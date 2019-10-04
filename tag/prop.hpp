/// prop.hpp
/// tag
///
/// Purpose:
/// Implement property tag
///

#include "tag/tag.hpp"

#ifndef TAG_PROP_HPP
#define TAG_PROP_HPP

namespace tag
{

/// Identifier for commutative property
const std::string commutative_tag = "commutative";

/// Identifier for immutable property
const std::string immutable_tag = "immutable";

/// PropTag (properties tag) define node properties
struct PropTag final : public iTag
{
	PropTag (std::string init_label) : labels_({init_label}) {}

	/// Implementation of iTag
	size_t tag_id (void) const override
	{
		return tag_id_;
	}

	/// Implementation of iTag
	void absorb (TagptrT&& other) override
	{
		std::unordered_set<std::string>& olabels =
			static_cast<PropTag*>(other.get())->labels_;
		labels_.insert(olabels.begin(), olabels.end());
	}

	/// Implementation of iTag
	TagRepsT get_tags (void) const override;

private:
	std::unordered_set<std::string> labels_;

	static size_t tag_id_;
};

/// TagRegistry wrapper to tag and check properties on tensors
struct PropertyRegistry final
{
	PropertyRegistry (TagRegistry& registry = get_reg()) :
		tag_reg_(registry) {}

	/// Associate property with tensor
	void property_tag (teq::TensrefT tens, std::string property)
	{
		tag_reg_.add_tag(tens, TagptrT(new PropTag(property)));
	}

	/// Return true if tensor has specified property
	bool has_property (const teq::iTensor* tens, std::string property) const;

	/// Internal tag registry used to retrieve tensor-property association
	TagRegistry& tag_reg_;
};

/// Return reference to global property registry
PropertyRegistry& get_property_reg (void);

/// Identifier of property tag
const std::string props_key = get_reg().register_tagr("properties",
[](teq::TensrefT ref, std::string property)
{
	get_property_reg().property_tag(ref, property);
});

}

#endif // TAG_PROP_HPP
