
#ifndef DISABLE_EIGEN_OBSERVABLE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mock.hpp"


TEST(OBSERVABLE, CopyMove)
{
	teq::Shape shape({3});
	auto lhs = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, shape);
	auto rhs = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5}, shape);

	marsh::Maps attrs;
	attrs.add_attr("dank", std::make_unique<marsh::Number<double>>(420));

	auto obs = std::make_shared<MockObservable>(
		std::move(attrs), teq::TensptrsT{lhs, rhs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"Hello", 1337});
	MockObservable parent(teq::TensptrsT{obs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"BigMama", 420});

	auto args = obs->get_args();
	auto parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(&parent, *parents.begin());

	MockObservable cpy(*obs);
	auto cargs = cpy.get_args();
	EXPECT_STREQ(obs->to_string().c_str(), cpy.to_string().c_str());
	EXPECT_VECEQ(args, cargs);
	EXPECT_EQ(0, cpy.get_observables().size());

	MockObservable cpass;
	cpass = cpy;
	auto caargs = cpass.get_args();
	EXPECT_STREQ(cpy.to_string().c_str(), cpass.to_string().c_str());
	EXPECT_VECEQ(args, caargs);
	EXPECT_EQ(0, cpass.get_observables().size());

	MockObservable mv(std::move(*obs));
	auto margs = mv.get_args();
	EXPECT_STREQ(cpy.to_string().c_str(), mv.to_string().c_str());
	EXPECT_VECEQ(args, margs);
	EXPECT_EQ(0, obs->get_observables().size());
	auto mvparents = mv.get_observables();
	ASSERT_EQ(1, mvparents.size());
	EXPECT_EQ(&parent, *mvparents.begin());

	MockObservable mvass;
	mvass = std::move(mv);
	auto maargs = mvass.get_args();
	EXPECT_STREQ(cpy.to_string().c_str(), mvass.to_string().c_str());
	EXPECT_VECEQ(args, maargs);
	EXPECT_EQ(0, mv.get_observables().size());
	auto maparents = mvass.get_observables();
	ASSERT_EQ(1, maparents.size());
	EXPECT_EQ(&parent, *maparents.begin());
}


TEST(OBSERVABLE, Subscriptions)
{
	teq::Shape shape({3});
	auto lhs = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, shape);
	auto rhs = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5}, shape);

	auto obs = std::make_shared<MockObservable>(teq::TensptrsT{lhs, rhs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"Hello", 1337});
	MockObservable parent(teq::TensptrsT{obs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"BigMama", 420});
	MockObservable unrelated(teq::TensptrsT{lhs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"Milkman", 132});

	auto parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(&parent, *parents.begin());

	obs->unsubscribe(&unrelated);

	parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(&parent, *parents.begin());

	obs->unsubscribe(&parent);

	parents = obs->get_observables();
	ASSERT_EQ(0, parents.size());

	obs->subscribe(&unrelated);

	parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(&unrelated, *parents.begin());
}


TEST(OBSERVABLE, Attribute)
{
	teq::Shape shape({3});
	auto lhs = std::make_shared<MockLeaf>(std::vector<double>{2, 8, 4}, shape);
	auto rhs = std::make_shared<MockLeaf>(std::vector<double>{3, 7, 5}, shape);

	marsh::Maps attrs;
	attrs.add_attr("dank", std::make_unique<marsh::Number<double>>(420));

	auto obs = std::make_shared<MockObservable>(
		std::move(attrs), teq::TensptrsT{lhs, rhs},
		std::vector<double>{1, 2, 3}, teq::Opcode{"Hello", 1337});

	auto keys = obs->ls_attrs();
	types::StringsT names = {"dank"};
	EXPECT_VECEQ(names, keys);

	auto num = dynamic_cast<marsh::Number<double>*>(
		obs->get_attr("dank"));
	EXPECT_NE(nullptr, num);
	EXPECT_EQ(420, num->to_int64());

	obs->add_attr("warm", std::make_unique<marsh::String>("colors"));

	[](const MockObservable& obs)
	{
		auto str = dynamic_cast<const marsh::String*>(
			obs.get_attr("warm"));
		EXPECT_NE(nullptr, str);
		EXPECT_STREQ("colors", str->to_string().c_str());
	}(*obs);

	EXPECT_EQ(2, obs->size());
	obs->rm_attr("dank");
	EXPECT_EQ(1, obs->size());
}


#endif // DISABLE_EIGEN_OBSERVABLE_TEST
