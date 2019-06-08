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

	void absorb (std::unique_ptr<iTag>&& other) override
	{
		std::vector<std::string>& olabels =
			static_cast<PropTag*>(other.get())->labels_;
		labels_.insert(labels_.end(),
			olabels.begin(), olabels.end());
		other.release();
	}

	TagRepsT get_tags (void) const override
	{
		TagRepsT out;
		out.emplace(props_key, labels_);
		return out;
	}

private:
	std::vector<std::string> labels_;

	static size_t tag_id_;
};

void property_tag (ade::TensrefT tens, std::string property);

}

#endif // TAG_PROP_HPP
