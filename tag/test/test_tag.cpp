
#ifndef DISABLE_TAG_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tag/tag.hpp"


struct MockTag : public tag::iTag
{
	size_t tag_id (void) const override
	{
		return tag_type_id;
	}

	void absorb (std::unique_ptr<tag::iTag>&& other) override
	{
		//
	}

	tag::TagRepsT get_tags (void) const override
	{
		return tag::TagRepsT();
	}

	size_t tag_type_id = 0;
};


TEST(TAG, MockRegistered)
{
	// since mock tag isn't registered...
}


#endif
