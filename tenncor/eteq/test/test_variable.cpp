
#ifndef DISABLE_ETEQ_VARIABLE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"


TEST(VARIABLE, Meta)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<float> big_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<int32_t> big_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::VarptrT<double> a(eteq::Variable<double>::get(big_d.data(), shape, "A"));
	eteq::VarptrT<float> b(eteq::Variable<float>::get(big_f.data(), shape, "B", teq::PLACEHOLDER));
	eteq::VarptrT<int32_t> c(eteq::Variable<int32_t>::get(big_i.data(), shape, "C", teq::IMMUTABLE));

	auto& ameta = a->get_meta();
	auto& bmeta = b->get_meta();
	auto& cmeta = c->get_meta();

	EXPECT_EQ(egen::DOUBLE, ameta.type_code());
	EXPECT_EQ(egen::FLOAT, bmeta.type_code());
	EXPECT_EQ(egen::INT32, cmeta.type_code());

	EXPECT_EQ(1, ameta.state_version());
	EXPECT_EQ(1, bmeta.state_version());
	EXPECT_EQ(1, cmeta.state_version());

	EXPECT_EQ(teq::VARUSAGE, a->get_usage());
	EXPECT_EQ(teq::PLACEHOLDER, b->get_usage());
	EXPECT_EQ(teq::IMMUTABLE, c->get_usage());

	EXPECT_STREQ("A", a->to_string().c_str());
	EXPECT_STREQ("B", b->to_string().c_str());
	EXPECT_STREQ("C", c->to_string().c_str());
}


TEST(VARIABLE, Assign)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<float> big_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<int32_t> big_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::EVariable<double> a(eteq::VarptrT<double>(
        eteq::Variable<double>::get(big_d.data(), shape, "A")));
	eteq::EVariable<float> b(eteq::VarptrT<float>(
        eteq::Variable<float>::get(big_f.data(), shape, "B")));
	eteq::EVariable<int32_t> c(eteq::VarptrT<int32_t>(
        eteq::Variable<int32_t>::get(big_i.data(), shape, "C")));

	std::vector<double> d = {3, 1, 222, 21, 17, 7, 91, 11, 71, 13, 81, 2};

	EXPECT_FATAL(a->assign(d.data(), teq::Shape({3, 7})),
		"assigning data shaped [3\\7\\1\\1\\1\\1\\1\\1] to tensor [3\\4\\1\\1\\1\\1\\1\\1]");
	a->assign(d.data(), teq::Shape({3, 4}));
	b->assign(d.data(), egen::DOUBLE, teq::Shape({3, 4}));
	c->assign(d.data(), egen::DOUBLE, teq::Shape({3, 4}));

	EXPECT_EQ(2, a->get_meta().state_version());
	EXPECT_EQ(3, b->get_meta().state_version());
	EXPECT_EQ(4, c->get_meta().state_version());
}


#endif // DISABLE_ETEQ_VARIABLE_TEST
