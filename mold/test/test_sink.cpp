#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp" 

#include "fuzzutil/fuzz.hpp"

#include "mold/inode.hpp"
#include "mold/sink.hpp"


#ifndef DISABLE_SINK_TEST


using namespace testutil;


class SINK : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		fuzz_test::TearDown();
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

protected:
	iNode* clone_impl (void) const override
	{
		return new mock_node(*this);
	}
};


TEST_F(SINK, Copy_E000)
{
	mock_node* bad = new mock_node();
	mock_node* node = new mock_node();
	mock_node* other = new mock_node();
	mold::Sink assign(other);
	mold::Sink assign2(other);
	mold::Sink sink(node);
	mold::Sink badsink(bad);
	delete bad;

	mold::Sink cp(sink);
	ASSERT_FALSE(cp.expired());
	EXPECT_EQ(node, cp.get());

	assign = sink;
	ASSERT_FALSE(assign.expired());
	EXPECT_EQ(node, assign.get());
	delete node;
	EXPECT_TRUE(cp.expired());
	EXPECT_TRUE(assign.expired());

	mold::Sink cp2(badsink);
	EXPECT_TRUE(cp2.expired());

	assign2 = badsink;
	EXPECT_TRUE(assign2.expired());
	delete other;
}


TEST_F(SINK, Move_E001)
{
	mock_node* bad = new mock_node();
	mock_node* node = new mock_node();
	mock_node* other = new mock_node();
	mold::Sink assign(other);
	mold::Sink assign2(other);
	mold::Sink sink(node);
	mold::Sink badsink(bad);
	delete bad;

	mold::Sink mv(std::move(sink));
	ASSERT_FALSE(mv.expired());
	EXPECT_EQ(node, mv.get());
	EXPECT_TRUE(sink.expired());

	assign = std::move(mv);
	ASSERT_FALSE(assign.expired());
	EXPECT_EQ(node, assign.get());
	EXPECT_TRUE(mv.expired());
	delete node;

	mold::Sink cp2(std::move(badsink));
	EXPECT_TRUE(cp2.expired());
	ASSERT_TRUE(badsink.expired());

	assign2 = std::move(badsink);
	EXPECT_TRUE(assign2.expired());
	EXPECT_TRUE(badsink.expired());
	delete other;
}


TEST_F(SINK, Assign_E002)
{
	mock_node* node = new mock_node();
	mock_node* other = new mock_node();
	mold::Sink sink(node);

	sink = other;
	auto aud = node->get_audience();
	EXPECT_TRUE(aud.empty());
	auto audoth = other->get_audience();
	EXPECT_FALSE(audoth.empty());
	EXPECT_EQ(other, sink.get());
	delete other;

	sink = node;
	aud = node->get_audience();
	EXPECT_FALSE(aud.empty());
	EXPECT_EQ(node, sink.get());
	delete node;
}


TEST_F(SINK, Expire_E003)
{
	mock_node* node = new mock_node();
	mold::Sink s(node);
	EXPECT_FALSE(s.expired());
	delete node;
	EXPECT_TRUE(s.expired());
}


#endif /* DISABLE_SINK_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
