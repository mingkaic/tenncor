
#ifndef DISABLE_ETEQ_VARIABLE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "tenncor/eteq/eteq.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


TEST(VARIABLE, CopyMove)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<float> big_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<int32_t> big_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::VarptrT a(eteq::Variable::get(big_d.data(), egen::DOUBLE, shape, "A"));
	eteq::VarptrT b(eteq::Variable::get(big_f.data(), egen::FLOAT, shape, "B"));
	eteq::VarptrT c(eteq::Variable::get(big_i.data(), egen::INT32, shape, "C"));

	EXPECT_STREQ("A", a->to_string().c_str());
	EXPECT_STREQ("B", b->to_string().c_str());
	EXPECT_STREQ("C", c->to_string().c_str());

	teq::TensptrT acpy(a->clone());
	teq::TensptrT bcpy(b->clone());
	teq::TensptrT ccpy(c->clone());

	EXPECT_STREQ("A", acpy->to_string().c_str());
	EXPECT_STREQ("B", bcpy->to_string().c_str());
	EXPECT_STREQ("C", ccpy->to_string().c_str());

	teq::TensptrT amv(a->move());
	teq::TensptrT bmv(b->move());
	teq::TensptrT cmv(c->move());

	EXPECT_STREQ("", a->to_string().c_str());
	EXPECT_STREQ("", b->to_string().c_str());
	EXPECT_STREQ("", c->to_string().c_str());
	EXPECT_STREQ("A", amv->to_string().c_str());
	EXPECT_STREQ("B", bmv->to_string().c_str());
	EXPECT_STREQ("C", cmv->to_string().c_str());
}


TEST(VARIABLE, Meta)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<float> big_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<int32_t> big_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::VarptrT a(eteq::Variable::get(big_d.data(), egen::DOUBLE, shape, "A"));
	eteq::VarptrT b(eteq::Variable::get(big_f.data(), egen::FLOAT, shape, "B", teq::PLACEHOLDER));
	eteq::VarptrT c(eteq::Variable::get(big_i.data(), egen::INT32, shape, "C", teq::IMMUTABLE));

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
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<float> big_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	std::vector<int32_t> big_i = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::EVariable a(eteq::VarptrT(
		eteq::Variable::get(big_d.data(), egen::DOUBLE, shape, "A")));
	eteq::EVariable b(eteq::VarptrT(
		eteq::Variable::get(big_f.data(), egen::FLOAT, shape, "B")));
	eteq::EVariable c(eteq::VarptrT(
		eteq::Variable::get(big_i.data(), egen::INT32, shape, "C")));

	std::vector<double> d = {3, 1, 222, 21, 17, 7, 91, 11, 71, 13, 81, 2};

	std::string fatalmsg = "assigning data shaped [3\\7] to tensor [3\\4]";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(a->assign(d.data(), teq::Shape({3, 7})), fatalmsg.c_str());
	a->assign(d.data(), teq::Shape({3, 4}));
	b->assign(d.data(), egen::DOUBLE, teq::Shape({3, 4}));
	c->assign(d.data(), egen::DOUBLE, teq::Shape({3, 4}));

	EXPECT_EQ(2, a->get_meta().state_version());
	EXPECT_EQ(3, b->get_meta().state_version());
	EXPECT_EQ(4, c->get_meta().state_version());

	eigen::TensorT<double> atensor(3, 4, 1, 1, 1, 1, 1, 1);
	atensor.setZero();
	a->assign(atensor);
	EXPECT_EQ(5, a->get_meta().state_version());

	auto adata = (double*) a->device().data();
	std::vector<double> zeros(12, 0);
	std::vector<double> avec(adata, adata + 12);
	EXPECT_VECEQ(zeros, avec);

	global::set_logger(new exam::NoSupportLogger());
}


#endif // DISABLE_ETEQ_VARIABLE_TEST
