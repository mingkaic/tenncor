#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp" 

#include "fuzzutil/fuzz.hpp"

#include "mold/inode.hpp"
#include "mold/sink.hpp"


#ifndef DISABLE_INODE_TEST


using namespace testutil;


class INODE : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutil::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


struct mock_node final : public mold::iNode, public testify::mocker
{
	bool has_data (void) const override
	{
		return true;
	}

	clay::State get_state (void) const override
	{
		return clay::State();
	}

	mold::iNode* derive (mold::iNode* wrt) override
	{
		return nullptr;
	}
};


TEST_F(INODE, CopyMove_A000)
{
	mock_node cpassign;
	mock_node mvassign;

	mock_node node;
	mold::Sink sink(&node);
	mold::Sink sink2(&node);

	auto aud = node.get_audience();
	ASSERT_EQ(2, aud.size());
	
	mock_node cp(node);
	auto cp_aud = cp.get_audience();
	EXPECT_EQ(0, cp_aud.size());
	EXPECT_EQ(&node, sink.get());
	EXPECT_EQ(&node, sink2.get());

	mock_node mv(std::move(node));
	auto aud2 = node.get_audience();
	ASSERT_EQ(0, aud2.size());
	auto mv_aud = mv.get_audience();
	EXPECT_EQ(2, mv_aud.size());
	EXPECT_EQ(&mv, sink.get());
	EXPECT_EQ(&mv, sink2.get());

	cpassign = cp;
	auto cpass_aud = cpassign.get_audience();
	EXPECT_EQ(0, cpass_aud.size());
	EXPECT_EQ(&mv, sink.get());
	EXPECT_EQ(&mv, sink2.get());

	mvassign = std::move(mv);
	auto mv_aud2 = mv.get_audience();
	ASSERT_EQ(0, mv_aud2.size());
	auto mvass_aud = mvassign.get_audience();
	EXPECT_EQ(2, mvass_aud.size());
	EXPECT_EQ(&mvassign, sink.get());
	EXPECT_EQ(&mvassign, sink2.get());
}


TEST_F(INODE, AudExpiration_A001)
{
	mock_node* node = new mock_node();
	mold::Sink sink(node);
	mold::Sink sink2(node);
	
	EXPECT_FALSE(sink.expired()) << "sink is expired before subject deletion";
	EXPECT_FALSE(sink2.expired()) << "sink2 is expired before subject deletion";
	delete node;

	EXPECT_TRUE(sink.expired()) << "sink is not expired after subject deletion";
	EXPECT_TRUE(sink2.expired()) << "sink2 is not expired after subject deletion";
}


#endif /* DISABLE_INODE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
