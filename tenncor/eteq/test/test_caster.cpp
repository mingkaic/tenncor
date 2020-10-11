
#ifndef DISABLE_ETEQ_CASTER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"


TEST(CASTER, Default)
{
	eigen::Packer<egen::_GENERATED_DTYPE> type_packer;
	eteq::TypeCaster<egen::ADD> caster;

	auto a = eteq::make_constant_scalar<double>(3, teq::Shape());
	auto b = eteq::make_constant_scalar<float>(3, teq::Shape());

	auto castchildren = caster.operator()<double>({a, b});
	// expect cast on b since b doesn't match operator type
	ASSERT_EQ(2, castchildren.size());
	EXPECT_STREQ("3", castchildren.front()->to_string().c_str());
	EXPECT_STREQ("CAST", castchildren.back()->to_string().c_str());

	EXPECT_EQ(egen::DOUBLE,
		castchildren.front()->get_meta().type_code());
	EXPECT_EQ(egen::DOUBLE,
		castchildren.back()->get_meta().type_code());

	auto bf = dynamic_cast<teq::iFunctor*>(castchildren.back().get());
	ASSERT_NE(nullptr, bf);

	egen::_GENERATED_DTYPE dtype = egen::BAD_TYPE;
	type_packer.unpack(dtype, *bf);
	EXPECT_EQ(egen::DOUBLE, dtype);

	auto castchildren2 = caster.operator()<float>({a, b}); // ensure order does not matter
	// expect cast on b
	ASSERT_EQ(2, castchildren2.size());
	EXPECT_STREQ("CAST", castchildren2.front()->to_string().c_str());
	EXPECT_STREQ("3", castchildren2.back()->to_string().c_str());

	EXPECT_EQ(egen::FLOAT,
		castchildren2.front()->get_meta().type_code());
	EXPECT_EQ(egen::FLOAT,
		castchildren2.back()->get_meta().type_code());

	auto af = dynamic_cast<teq::iFunctor*>(castchildren2.front().get());
	ASSERT_NE(nullptr, af);

	dtype = egen::BAD_TYPE;
	type_packer.unpack(dtype, *af);
	EXPECT_EQ(egen::FLOAT, dtype);

	auto castchildren3 = caster.operator()<int32_t>({a, b}); // ensure both can be casted
	// expect cast on a and b
	ASSERT_EQ(2, castchildren3.size());
	EXPECT_STREQ("CAST", castchildren3.front()->to_string().c_str());
	EXPECT_STREQ("CAST", castchildren3.back()->to_string().c_str());

	auto af2 = dynamic_cast<teq::iFunctor*>(castchildren3.front().get());
	auto bf2 = dynamic_cast<teq::iFunctor*>(castchildren3.back().get());
	ASSERT_NE(nullptr, af2);
	ASSERT_NE(nullptr, bf2);

	dtype = egen::BAD_TYPE;
	type_packer.unpack(dtype, *af2);
	EXPECT_EQ(egen::INT32, dtype);

	dtype = egen::BAD_TYPE;
	type_packer.unpack(dtype, *bf2);
	EXPECT_EQ(egen::INT32, dtype);

	EXPECT_EQ(egen::INT32,
		castchildren3.front()->get_meta().type_code());
	EXPECT_EQ(egen::INT32,
		castchildren3.back()->get_meta().type_code());
}


TEST(CASTER, Cast)
{
	eteq::TypeCaster<egen::CAST> caster;

	auto a = eteq::make_constant_scalar<float>(3, teq::Shape());

	auto castchildren = caster.operator()<int32_t>({a});
	// expect a to never be casted
	ASSERT_EQ(1, castchildren.size());
	EXPECT_STREQ("3", castchildren.front()->to_string().c_str());
	EXPECT_EQ(egen::FLOAT,
		castchildren.front()->get_meta().type_code());
}


#endif // DISABLE_ETEQ_CASTER_TEST
