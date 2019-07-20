
#ifndef DISABLE_GROUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tag/test/common.hpp"

#include "tag/group.hpp"


TEST(GROUP, SingleTagAdjacency)
{
    tag::TagRegistry treg;
    tag::GroupRegistry registry(treg);
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


TEST(GROUP, RecursiveTagAdjacency)
{
    tag::TagRegistry treg;
    tag::GroupRegistry registry(treg);
	{
		ade::TensptrT tens = std::make_shared<MockTensor>();
        ade::TensptrT tens2 = std::make_shared<MockTensor>();
        ade::TensptrT f(ade::Functor::get(ade::Opcode{"MOCK", 2}, {
            ade::identity_map(tens),
    		ade::identity_map(tens2),
        }));

        tag::recursive_group_tag(tens, "group2",
            std::unordered_set<ade::iTensor*>{},
            registry);
        tag::recursive_group_tag(tens2, "group2",
            std::unordered_set<ade::iTensor*>{},
            registry);
        tag::recursive_group_tag(tens, "group7",
            std::unordered_set<ade::iTensor*>{},
            registry);
        tag::recursive_group_tag(f, "group1",
            std::unordered_set<ade::iTensor*>{},
            registry);
        tag::recursive_group_tag(f, "group3",
            std::unordered_set<ade::iTensor*>{tens.get()},
            registry);
        tag::recursive_group_tag(f, "group4",
            std::unordered_set<ade::iTensor*>{tens.get(), tens2.get()},
            registry);

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
        EXPECT_EQ(3, t2groups.size());
        EXPECT_EQ(3, fgroups.size());

        ASSERT_HAS(tgroups, "group7");
        ASSERT_HAS(tgroups, "group2");
        ASSERT_HAS(tgroups, "group1");

        ASSERT_HAS(t2groups, "group2");
        ASSERT_HAS(t2groups, "group1");
        ASSERT_HAS(t2groups, "group3");

        ASSERT_HAS(fgroups, "group1");
        ASSERT_HAS(fgroups, "group3");
        ASSERT_HAS(fgroups, "group4");

        // check for adjacency
        EXPECT_EQ(tgroups["group1"], t2groups["group1"]);
        EXPECT_EQ(tgroups["group1"], fgroups["group1"]);

        EXPECT_EQ(fgroups["group3"], t2groups["group3"]);

        // not adjacent, even though same group
        EXPECT_NE(tgroups["group2"], t2groups["group2"]);

        // each group is different from each other
        EXPECT_NE(tgroups["group7"], tgroups["group2"]);
        EXPECT_NE(tgroups["group7"], tgroups["group1"]);
        EXPECT_NE(tgroups["group7"], t2groups["group2"]);
        EXPECT_NE(tgroups["group7"], t2groups["group3"]);
        EXPECT_NE(tgroups["group7"], fgroups["group4"]);
    }
}


TEST(GROUP, Subgraph)
{
    tag::TagRegistry treg;
    tag::GroupRegistry registry(treg);
	{
		ade::TensptrT tens = std::make_shared<MockTensor>();
        ade::TensptrT tens2 = std::make_shared<MockTensor>();
        ade::TensptrT f(ade::Functor::get(ade::Opcode{"MOCK", 2}, {
            ade::identity_map(tens),
    		ade::identity_map(tens2),
        }));

        tag::AdjacentGroups adjer(registry);
        adjer.visited_ = {f.get(), tens.get(), tens2.get()};
        adjer.adjs_ = std::unordered_map<ade::iTensor*,tag::AGroupsT>{
            std::pair<ade::iTensor*,tag::AGroupsT>{f.get(),
            tag::AGroupsT{
                {"group1", {"lytening"}},
                {"group3", {"fyreball"}},
                {"group4", {"frostbyte"}},
            }},
            std::pair<ade::iTensor*,tag::AGroupsT>{tens.get(),
            tag::AGroupsT{
                {"group7", {"kaostorm"}},
                {"group2", {"mudslyde"}},
                {"group1", {"lytening"}},
            }},
            std::pair<ade::iTensor*,tag::AGroupsT>{tens2.get(),
            tag::AGroupsT{
                {"group2", {"sandstrum"}},
                {"group1", {"lytening"}},
                {"group3", {"fyreball"}},
            }}
        };

        tag::SubgraphAssocsT assocs;
        tag::beautify_groups(assocs, adjer);
        tag::filter_head(assocs, assocs);

        EXPECT_EQ(3, assocs.size());
        ASSERT_HAS(assocs, f.get());
        ASSERT_HAS(assocs, tens.get());
        ASSERT_HAS(assocs, tens2.get());

        {
            auto& sgs = assocs[f.get()]; // associations where f is the head of
            std::unordered_set<std::string> groups;
            std::transform(sgs.begin(), sgs.end(), std::inserter(groups, groups.begin()),
                [](tag::SgraphptrT sg)
                {
                    return sg->group_;
                });
            EXPECT_EQ(3, groups.size());
            EXPECT_HAS(groups, "group1");
            EXPECT_HAS(groups, "group3");
            EXPECT_HAS(groups, "group4");
        }

        {
            auto& sgs = assocs[tens.get()]; // associations where tens is the head of
            std::unordered_set<std::string> groups;
            std::transform(sgs.begin(), sgs.end(), std::inserter(groups, groups.begin()),
                [](tag::SgraphptrT sg)
                {
                    return sg->group_;
                });
            EXPECT_EQ(2, groups.size());
            EXPECT_HAS(groups, "group7");
            EXPECT_HAS(groups, "group2");
        }

        {
            auto& sgs = assocs[tens2.get()]; // associations where tens2 is the head of
            std::unordered_set<std::string> groups;
            std::transform(sgs.begin(), sgs.end(), std::inserter(groups, groups.begin()),
                [](tag::SgraphptrT sg)
                {
                    return sg->group_;
                });
            EXPECT_EQ(1, groups.size());
            EXPECT_HAS(groups, "group2");
        }
    }
}


#endif // DISABLE_GROUP_TEST
