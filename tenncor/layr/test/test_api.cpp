
#ifndef DISABLE_LAYER_API_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/layr/layer.hpp"


TEST(LAYER, MakeGetInput)
{
	teq::Shape shape({3, 2});
	teq::TensptrT x = std::make_shared<MockLeaf>(
		std::vector<double>{}, shape, "x");
	teq::TensptrT x2 = std::make_shared<MockLeaf>(
		std::vector<double>{}, shape, "x2");
	teq::TensptrT x3 = std::make_shared<MockLeaf>(
		std::vector<double>{}, shape, "x3");

	auto layer_root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{
			x, x2
		}, teq::Opcode{"ADD", 5}),
		x3
	}, teq::Opcode{"MUL", 6});
	auto layer = layr::make_layer(layer_root, "example", x);
	EXPECT_EQ(x.get(), layr::get_input(layer).get());
}


#endif // DISABLE_LAYER_API_TEST
