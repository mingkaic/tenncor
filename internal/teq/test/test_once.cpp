
#ifndef DISABLE_TEQ_ONCE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/teq.hpp"


TEST(ONCE, Scoping)
{
	bool done = false;
	{
		teq::Once<size_t> val(2, [&done]{ done = true; });
		EXPECT_EQ(2, val.get());
	}
	EXPECT_TRUE(done);
}


TEST(ONCE, MoveCreation)
{
	size_t called = 0;
	{
		teq::Once<size_t> first(2, [&called]{ ++called; });
		{
			teq::Once<size_t> val(3, std::move(first));
			EXPECT_EQ(3, val.get());
			EXPECT_EQ(0, called);
		}
		EXPECT_EQ(2, first.get());
		EXPECT_EQ(1, called);
	}
	EXPECT_EQ(1, called);
}


TEST(ONCE, Moving)
{
	bool init = false;
	bool inner = false;
	{
		teq::Once<size_t> first(2, [&init]{ init = true; });
		EXPECT_EQ(2, first.get());
		{
			teq::Once<size_t> val(1, [&inner]{ inner = true; });
			EXPECT_EQ(1, val.get());

			first = std::move(val);
			EXPECT_EQ(1, val.get());
			EXPECT_EQ(1, first.get());
			EXPECT_TRUE(init);
			EXPECT_FALSE(inner);
		}
		EXPECT_TRUE(init);
		EXPECT_FALSE(inner);
	}
	EXPECT_TRUE(init);
	EXPECT_TRUE(init);
}


#endif // DISABLE_TEQ_ONCE_TEST
