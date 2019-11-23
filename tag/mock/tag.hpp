#include "tag/tag.hpp"

const std::string tag_key = "mock_tag";

const std::vector<std::string> tag_values{"A", "B", "C"};

static size_t tid = 1234;

struct MockTag final : public tag::iTag
{
	MockTag (void) = default;

	MockTag (size_t tagid, std::string tagk) :
		tid_(tagid), tag_key_(tagk) {}

	size_t tag_id (void) const override
	{
		return tid_;
	}

	void absorb (std::unique_ptr<tag::iTag>&& other) override
	{
		++add_count_;
	}

	tag::TagRepsT get_tags (void) const override
	{
		std::vector<std::string> tag_values_cpy = tag_values;
		tag_values_cpy.push_back(fmts::to_string(add_count_));
		return tag::TagRepsT({
			{tag_key_, tag_values_cpy},
		});
	}

	size_t add_count_ = 1;

	size_t tid_ = tid;

	std::string tag_key_ = tag_key;
};
