
#ifndef DISABLE_TRAVELER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "teq/traveler.hpp"


TEST(TRAVELER, GraphStat)
{
	teq::TensptrT a(new MockLeaf({}, teq::Shape()));
	teq::TensptrT b(new MockLeaf({}, teq::Shape()));
	teq::TensptrT c(new MockLeaf({}, teq::Shape()));

	teq::TensptrT f(new MockFunctor(teq::TensptrsT{a, b},
		std::vector<double>{}, teq::Opcode{"MOCK1", 1}));

	teq::TensptrT g(new MockFunctor(teq::TensptrsT{c, f},
		std::vector<double>{}, teq::Opcode{"MOCK0", 0}));

	teq::GraphStat stat;
	g->accept(stat);
	EXPECT_EQ(2, stat.graphsize_[g.get()].upper_);
	EXPECT_EQ(1, stat.graphsize_[f.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[c.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[a.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[b.get()].upper_);
}


TEST(TRAVELER, PathFinder)
{
	teq::TensptrT a(new MockLeaf({}, teq::Shape()));
	teq::TensptrT b(new MockLeaf({}, teq::Shape()));
	teq::TensptrT c(new MockLeaf({}, teq::Shape()));

	teq::TensptrT f(new MockFunctor(teq::TensptrsT{a, b},
		std::vector<double>{}, teq::Opcode{"MOCK1", 1}));

	teq::TensptrT g(new MockFunctor(teq::TensptrsT{c, f},
		std::vector<double>{}, teq::Opcode{"MOCK1", 1}));

	teq::PathFinder finder(a.get());
	g->accept(finder);

	{
		ASSERT_HAS(finder.roadmap_, g.get());
		EXPECT_ARRHAS(finder.roadmap_[g.get()], 1);

		ASSERT_HAS(finder.roadmap_, f.get());
		EXPECT_ARRHAS(finder.roadmap_[f.get()], 0);
	}

	finder.clear();
	f->accept(finder);

	{
		ASSERT_HASNOT(finder.roadmap_, g.get());

		ASSERT_HAS(finder.roadmap_, f.get());
		EXPECT_ARRHAS(finder.roadmap_[f.get()], 0);
	}

	teq::PathFinder finder2(c.get());
	g->accept(finder2);

	{
		ASSERT_HAS(finder2.roadmap_, g.get());
		EXPECT_ARRHAS(finder2.roadmap_[g.get()], 0);
	}

	finder2.clear();
	f->accept(finder2);

	EXPECT_HASNOT(finder2.roadmap_, f.get());
	EXPECT_EQ(0, finder2.roadmap_.size());
}


TEST(TRAVELER, ReverseParentGraph)
{
	teq::TensptrT a(new MockLeaf({}, teq::Shape()));
	teq::TensptrT b(new MockLeaf({}, teq::Shape()));
	teq::TensptrT c(new MockLeaf({}, teq::Shape()));

	teq::TensptrT f(new MockFunctor(teq::TensptrsT{a, b},
		std::vector<double>{}, teq::Opcode{"f", 1}));

	teq::TensptrT g(new MockFunctor(teq::TensptrsT{f, b},
		std::vector<double>{}, teq::Opcode{"g", 2}));

	teq::TensptrT h(new MockFunctor(teq::TensptrsT{c, f, g},
		std::vector<double>{}, teq::Opcode{"h", 3}));

	teq::ParentFinder finder;
	h->accept(finder);

	// expect: a -> [f], b -> [f, g], c -> [h], f -> [g, h], g -> [h], h -> []
	auto& parents = finder.parents_;
	auto aparents = parents[a.get()];
	auto bparents = parents[b.get()];
	auto cparents = parents[c.get()];
	auto fparents = parents[f.get()];
	auto gparents = parents[g.get()];
	auto hparents = parents[h.get()];

	EXPECT_EQ(1, aparents.size());
	EXPECT_EQ(2, bparents.size());
	EXPECT_EQ(1, cparents.size());
	EXPECT_EQ(2, fparents.size());
	EXPECT_EQ(1, gparents.size());
	EXPECT_EQ(0, hparents.size());

	EXPECT_HAS(aparents, f.get());
	EXPECT_HAS(bparents, f.get());
	EXPECT_HAS(bparents, g.get());
	EXPECT_HAS(cparents, h.get());
	EXPECT_HAS(fparents, g.get());
	EXPECT_HAS(fparents, h.get());
	EXPECT_HAS(gparents, h.get());
}


TEST(TRAVELER, Owners)
{
	teq::OwnerMapT owners;
	teq::TensptrT a(new MockLeaf({}, teq::Shape()));
	teq::TensptrT b(new MockLeaf({}, teq::Shape()));
	teq::TensptrT c(new MockLeaf({}, teq::Shape()));
	teq::iTensor* fref;
	teq::iTensor* gref;
	{
		teq::TensptrT f(new MockFunctor(teq::TensptrsT{a, b},
			std::vector<double>{}, teq::Opcode{"f", 1}));

		teq::TensptrT g(new MockFunctor(teq::TensptrsT{f, c},
			std::vector<double>{}, teq::Opcode{"g", 2}));
		fref = f.get();
		gref = g.get();

		owners = teq::track_owners({g});
		ASSERT_HAS(owners, a.get());
		ASSERT_HAS(owners, b.get());
		ASSERT_HAS(owners, c.get());
		ASSERT_HAS(owners, fref);
		ASSERT_HAS(owners, gref);

		EXPECT_FALSE(owners[a.get()].expired());
		EXPECT_FALSE(owners[b.get()].expired());
		EXPECT_FALSE(owners[c.get()].expired());
		EXPECT_FALSE(owners[fref].expired());
		EXPECT_FALSE(owners[gref].expired());

		auto alocked = owners[a.get()].lock();
		auto blocked = owners[b.get()].lock();
		auto clocked = owners[c.get()].lock();
		auto flocked = owners[fref].lock();
		auto glocked = owners[gref].lock();
		EXPECT_EQ(a.use_count(), alocked.use_count());
		EXPECT_EQ(b.use_count(), blocked.use_count());
		EXPECT_EQ(c.use_count(), clocked.use_count());
		EXPECT_EQ(f.use_count(), flocked.use_count());
		EXPECT_EQ(g.use_count(), glocked.use_count());
	}

	EXPECT_FALSE(owners[a.get()].expired());
	EXPECT_FALSE(owners[b.get()].expired());
	EXPECT_FALSE(owners[c.get()].expired());
	EXPECT_TRUE(owners[fref].expired());
	EXPECT_TRUE(owners[gref].expired());
}


#endif // DISABLE_TRAVELER_TEST
