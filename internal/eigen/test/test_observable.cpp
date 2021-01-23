
#ifndef DISABLE_EIGEN_OBSERVABLE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mock.hpp"


TEST(OBSERVABLE, CopyMove)
{
	auto obs = std::make_shared<MockMObservable>();
	auto parent = make_obs("", 0, teq::TensptrsT{obs});

	auto parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(parent.get(), *parents.begin());

	MockMObservable cpy(*obs);
	EXPECT_EQ(0, cpy.get_observables().size());

	MockMObservable cpass;
	cpass = cpy;
	EXPECT_EQ(0, cpass.get_observables().size());

	MockMObservable mv(std::move(*obs));
	EXPECT_EQ(0, obs->get_observables().size());
	auto mvparents = mv.get_observables();
	ASSERT_EQ(1, mvparents.size());
	EXPECT_EQ(parent.get(), *mvparents.begin());

	MockMObservable mvass;
	mvass = std::move(mv);
	EXPECT_EQ(0, mv.get_observables().size());
	auto maparents = mvass.get_observables();
	ASSERT_EQ(1, maparents.size());
	EXPECT_EQ(parent.get(), *maparents.begin());
}


TEST(OBSERVABLE, Subscriptions)
{
	auto lhs = std::make_shared<MockLeaf>();

	auto obs = std::make_shared<MockMObservable>();
	auto parent = make_obs("", 0, teq::TensptrsT{obs});
	auto unrelated = make_obs("", 0, teq::TensptrsT{lhs});

	auto parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(parent.get(), *parents.begin());

	obs->unsubscribe(unrelated.get());

	parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(parent.get(), *parents.begin());

	obs->unsubscribe(parent.get());

	parents = obs->get_observables();
	ASSERT_EQ(0, parents.size());

	obs->subscribe(unrelated.get());

	parents = obs->get_observables();
	ASSERT_EQ(1, parents.size());
	EXPECT_EQ(unrelated.get(), *parents.begin());
}


TEST(OBSERVABLE, Attribute)
{
	auto lhs = std::make_shared<MockLeaf>();
	auto rhs = std::make_shared<MockLeaf>();

	marsh::Maps attrs;
	attrs.add_attr("dank", std::make_unique<marsh::Number<double>>(420));

	auto obs = make_obs("", 0, teq::TensptrsT{lhs, rhs}, std::move(attrs));

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
