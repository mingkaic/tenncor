#include "gtest/gtest.h"

#include "util/rand.hpp"


#ifndef DISABLE_RAND_TEST


TEST(RAND, UID)
{
	int thing;
	std::string uid = make_uid(&thing);
	std::string uid2 = make_uid(&thing);
	EXPECT_STRNE(uid.c_str(), uid2.c_str());
}


#endif /* DISABLE_RAND_TEST */
