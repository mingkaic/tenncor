
#ifndef DISABLE_TRAVELER_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ade/test/common.hpp"

#include "ade/functor.hpp"
#include "ade/traveler.hpp"


struct TRAVELER : public ::testing::Test
{
	virtual void TearDown (void)
	{
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(TRAVELER, GraphStat)
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
	EXPECT_EQ(2, stat.graphsize_[g.get()]);
	EXPECT_EQ(1, stat.graphsize_[f.get()]);
	EXPECT_EQ(0, stat.graphsize_[c.get()]);
	EXPECT_EQ(0, stat.graphsize_[a.get()]);
	EXPECT_EQ(0, stat.graphsize_[b.get()]);
}


TEST_F(TRAVELER, PathFinder)
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
		auto it = finder.parents_.find(g.get());
		ASSERT_TRUE(finder.parents_.end() != it);
		EXPECT_TRUE(it->second.end() != it->second.find(1));

		it = finder.parents_.find(f.get());
		ASSERT_TRUE(finder.parents_.end() != it);
		EXPECT_TRUE(it->second.end() != it->second.find(0));
	}

	finder.parents_.clear();
	f->accept(finder);

	{
		ASSERT_TRUE(finder.parents_.end() == finder.parents_.find(g.get()));

		auto it = finder.parents_.find(f.get());
		ASSERT_TRUE(finder.parents_.end() != it);
		EXPECT_TRUE(it->second.end() != it->second.find(0));
	}

	ade::PathFinder finder2(c.get());
	g->accept(finder2);

	{
		auto it = finder2.parents_.find(g.get());
		ASSERT_TRUE(finder2.parents_.end() != it);
		EXPECT_TRUE(it->second.end() != it->second.find(0));
	}

	finder2.parents_.clear();
	f->accept(finder2);

	ASSERT_TRUE(finder2.parents_.end() == finder2.parents_.find(f.get()));
	EXPECT_EQ(0, finder2.parents_.size());
}


#endif // DISABLE_TRAVELER_TEST
