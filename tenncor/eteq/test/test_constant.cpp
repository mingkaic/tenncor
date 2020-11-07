
#ifndef DISABLE_ETEQ_CONSTANT_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"


TEST(CONSTANT, Copy)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	auto a = eteq::Constant<double>::get(big_d.data(), shape);
	EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", a->to_string().c_str());

	auto cpy = a->clone();
	EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", cpy->to_string().c_str());

	delete a;
	delete cpy;
}


TEST(CONSTANT, Meta)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	auto a = eteq::Constant<double>::get(big_d.data(), shape);

	auto& meta = a->get_meta();

	EXPECT_EQ(egen::DOUBLE, meta.type_code());

	EXPECT_EQ(1, meta.state_version());

	EXPECT_EQ(teq::IMMUTABLE, a->get_usage());

	EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", a->to_string().c_str());

	auto ashape = a->shape();
	ASSERT_ARREQ(shape, ashape);

	auto adata = (double*) a->device().data();
	std::vector<double> avec(adata, adata + shape.n_elems());
	EXPECT_ARREQ(big_d, avec);

	delete a;
}


#endif // DISABLE_ETEQ_CONSTANT_TEST
