
#ifndef DISABLE_TRAVELER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ade/test/common.hpp"

#include "ade/functor.hpp"
#include "ade/traveler.hpp"


TEST(TRAVELER, GraphStat)
{
	ade::TensptrT a(new MockTensor());
	ade::TensptrT b(new MockTensor());
	ade::TensptrT c(new MockTensor());

	ade::TensptrT f(ade::Functor::get(ade::Opcode{"MOCK1", 1}, {
		ade::identity_map(a),
		ade::identity_map(b),
	}));

	ade::TensptrT g(ade::Functor::get(ade::Opcode{"MOCK0", 0}, {
		ade::identity_map(c),
		ade::identity_map(f),
	}));

	ade::GraphStat stat;
	g->accept(stat);
	EXPECT_EQ(2, stat.graphsize_[g.get()].upper_);
	EXPECT_EQ(1, stat.graphsize_[f.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[c.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[a.get()].upper_);
	EXPECT_EQ(0, stat.graphsize_[b.get()].upper_);
}


TEST(TRAVELER, PathFinder)
{
	ade::TensptrT a(new MockTensor());
	ade::TensptrT b(new MockTensor());
	ade::TensptrT c(new MockTensor());

	ade::TensptrT f(ade::Functor::get(ade::Opcode{"MOCK1", 1}, {
		ade::identity_map(a),
		ade::identity_map(b),
	}));

	ade::TensptrT g(ade::Functor::get(ade::Opcode{"MOCK1", 1}, {
		ade::identity_map(c),
		ade::identity_map(f),
	}));

	ade::PathFinder finder(a.get());
	g->accept(finder);

	{
		ASSERT_HAS(finder.parents_, g.get());
		EXPECT_HAS(finder.parents_[g.get()], 1);

		ASSERT_HAS(finder.parents_, f.get());
		EXPECT_HAS(finder.parents_[f.get()], 0);
	}

	finder.parents_.clear();
	f->accept(finder);

	{
		ASSERT_HASNOT(finder.parents_, g.get());

		ASSERT_HAS(finder.parents_, f.get());
		EXPECT_HAS(finder.parents_[f.get()], 0);
	}

	ade::PathFinder finder2(c.get());
	g->accept(finder2);

	{
		ASSERT_HAS(finder2.parents_, g.get());
		EXPECT_HAS(finder2.parents_[g.get()], 0);
	}

	finder2.parents_.clear();
	f->accept(finder2);

	EXPECT_HASNOT(finder2.parents_, f.get());
	EXPECT_EQ(0, finder2.parents_.size());
}


TEST(TRAVELER, ReverseParentGraph)
{
	ade::TensptrT a(new MockTensor());
	ade::TensptrT b(new MockTensor());
	ade::TensptrT c(new MockTensor());

	ade::TensptrT f(ade::Functor::get(ade::Opcode{"f", 1}, {
		ade::identity_map(a),
		ade::identity_map(b),
	}));

	ade::TensptrT g(ade::Functor::get(ade::Opcode{"g", 2}, {
		ade::identity_map(f),
		ade::identity_map(b),
	}));

	ade::TensptrT h(ade::Functor::get(ade::Opcode{"h", 3}, {
		ade::identity_map(c),
		ade::identity_map(f),
		ade::identity_map(g),
	}));

	ade::ParentFinder finder;
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


#endif // DISABLE_TRAVELER_TEST
