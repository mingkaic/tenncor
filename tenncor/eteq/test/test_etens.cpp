
#ifndef DISABLE_ETEQ_ETENS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"


TEST(ETENS, ETensRegistry)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::ETensor a(teq::TensptrT(eteq::Constant<double>::get(big_d.data(), shape)));

	auto& registry = eteq::get_reg();
	EXPECT_EQ(1, registry.size());
}


TEST(ETENS, EVarRegistry)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::EVariable<double> a(eteq::VarptrT<double>(
		eteq::Variable<double>::get(big_d.data(), shape, "A")));

	auto& registry = eteq::get_reg();
	EXPECT_EQ(1, registry.size());
}


#endif // DISABLE_ETEQ_ETENS_TEST
