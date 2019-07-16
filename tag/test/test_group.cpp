
#ifndef DISABLE_GROUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tag/test/common.hpp"

#include "tag/group.hpp"


TEST(GROUP, Tag)
{
    tag::TagRegistry treg;
    tag::GroupRegistry registry(treg);
	ade::iTensor* ptr;
	{
		ade::TensptrT tens = std::make_shared<MockTensor>();
        registry.group_tag(tens, "group2");
        registry.group_tag(tens, "group7");
        registry.group_tag(tens, "group1");

        ptr = tens.get();
    }
}


#endif // DISABLE_GROUP_TEST
