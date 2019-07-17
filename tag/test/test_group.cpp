
#ifndef DISABLE_GROUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tag/test/common.hpp"

#include "tag/group.hpp"


TEST(GROUP, SingleTagAdjacency)
{
    tag::TagRegistry treg;
    tag::GroupRegistry registry(treg);
	ade::iTensor* ptr;
	{
		ade::TensptrT tens = std::make_shared<MockTensor>();
        ade::TensptrT tens2 = std::make_shared<MockTensor>();
        ade::TensptrT f(ade::Functor::get(ade::Opcode{"MOCK", 2}, {
            ade::identity_map(tens),
    		ade::identity_map(tens2),
        }));

        registry.group_tag(tens, "group2");
        registry.group_tag(tens, "group7");
        registry.group_tag(tens, "group1");

        registry.group_tag(tens2, "group2");
        registry.group_tag(tens2, "group1");

        registry.group_tag(f, "group1");

        tag::AdjacentGroups adjer(registry);
        tens->accept(adjer);
        tens2->accept(adjer);
        f->accept(adjer);

        ASSERT_HAS(adjer.adjs_, tens.get());
        ASSERT_HAS(adjer.adjs_, tens2.get());
        ASSERT_HAS(adjer.adjs_, f.get());

        auto& tgroups = adjer.adjs_[tens.get()];
        auto& t2groups = adjer.adjs_[tens2.get()];
        auto& fgroups = adjer.adjs_[f.get()];

        // check groups are registered
        EXPECT_EQ(3, tgroups.size());
        EXPECT_EQ(2, t2groups.size());
        EXPECT_EQ(1, fgroups.size());

        ASSERT_HAS(tgroups, "group7");
        ASSERT_HAS(tgroups, "group2");
        ASSERT_HAS(tgroups, "group1");

        ASSERT_HAS(t2groups, "group2");
        ASSERT_HAS(t2groups, "group1");

        ASSERT_HAS(fgroups, "group1");

        // check for adjacency
        EXPECT_EQ(tgroups["group1"], t2groups["group1"]);
        EXPECT_EQ(tgroups["group1"], fgroups["group1"]);

        // not adjacent, even though same group
        EXPECT_NE(tgroups["group2"], t2groups["group2"]);

        // each group is different from each other
        EXPECT_NE(tgroups["group7"], tgroups["group2"]);
        EXPECT_NE(tgroups["group7"], tgroups["group1"]);
        EXPECT_NE(tgroups["group7"], t2groups["group2"]);
    }
}


#endif // DISABLE_GROUP_TEST
