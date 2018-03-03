#include "gtest/gtest.h"

#include "utils/utils.hpp"

#include "fuzz/irng.hpp"

int main(int argc, char **argv) {
	testify::set_generator(nnutils::get_generator);
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
