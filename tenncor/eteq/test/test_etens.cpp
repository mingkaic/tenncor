
#ifndef DISABLE_ETEQ_ETENS_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/eteq/eteq.hpp"


TEST(ETENS, CopyMove)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::ETensor a(teq::TensptrT(eteq::Constant<double>::get(big_d.data(), shape)));

	eteq::ETensor cpy(a);
	EXPECT_EQ(a.get(), cpy.get());

	eteq::ETensor ass;
	ass = a;
	EXPECT_EQ(a.get(), ass.get());

	auto& registry = eteq::get_reg();
	EXPECT_EQ(3, registry.size());

	eteq::ETensor mv(std::move(a));
	EXPECT_EQ(nullptr, a.get());
	EXPECT_EQ(cpy.get(), mv.get());

	eteq::ETensor mvass;
	mvass = std::move(mv);
	EXPECT_EQ(nullptr, mv.get());
	EXPECT_EQ(cpy.get(), mvass.get());

	EXPECT_EQ(3, registry.size());
}


TEST(ETENS, NullCheck)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});

	eteq::ETensor a(teq::TensptrT(eteq::Constant<double>::get(big_d.data(), shape)));
	eteq::ETensor b;

	EXPECT_FALSE(a == nullptr);
	EXPECT_FALSE(nullptr == a);
	EXPECT_TRUE(a != nullptr);
	EXPECT_TRUE(nullptr != a);

	EXPECT_TRUE(b == nullptr);
	EXPECT_TRUE(nullptr == b);
	EXPECT_FALSE(b != nullptr);
	EXPECT_FALSE(nullptr != b);
}


TEST(ETENS, Get)
{
	eteq::ETensor aglobal;
	{
		std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
		teq::Shape shape({3, 4});

		teq::TensptrT atens(eteq::Constant<double>::get(big_d.data(), shape));

		auto etens_ctx = std::make_shared<estd::ConfigMap<>>();
		eteq::ETensor a(atens, etens_ctx);
		eteq::ETensor b;

		auto ac = atens.get();
		auto acontent = a.get();
		auto bcontent = b.get();

		EXPECT_EQ(ac, acontent);
		EXPECT_EQ(nullptr, bcontent);

		EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", a->to_string().c_str());
		EXPECT_STREQ("[1\\2\\3\\4\\5\\...]", (*a).to_string().c_str());

		EXPECT_EQ(etens_ctx, a.get_context());
		EXPECT_EQ(nullptr, b.get_context());

		aglobal = a;
		EXPECT_EQ(etens_ctx, aglobal.get_context());
	}
	EXPECT_EQ(nullptr, aglobal.get_context());
}


TEST(ETENS, Convert)
{
	std::vector<double> big_d = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	teq::Shape shape({3, 4});
	teq::TensptrT atens(eteq::Constant<double>::get(big_d.data(), shape));
	teq::TensptrT ctens(eteq::Constant<double>::get(big_d.data(), shape));

	eteq::ETensor a(atens);
	eteq::ETensor b;
	eteq::ETensor c(ctens);

	teq::TensptrT aconv = a;
	teq::TensptrT bconv = b;
	teq::TensptrT cconv = c;

	eteq::ETensorsT args{a, b, c};
	auto convargs = eteq::to_tensors(args);

	EXPECT_EQ(atens, aconv);
	EXPECT_EQ(nullptr, bconv);
	EXPECT_EQ(ctens, cconv);

	ASSERT_EQ(3, convargs.size());
	EXPECT_EQ(atens, convargs[0]);
	EXPECT_EQ(nullptr, convargs[1]);
	EXPECT_EQ(ctens, convargs[2]);
}


TEST(ETENS, DataCalc)
{
	teq::Shape shape;
	double data = 3;

	eteq::EVariable<double> a(eteq::VarptrT<double>(
		eteq::Variable<double>::get(&data, shape, "A")));
	eteq::EVariable<double> b(eteq::VarptrT<double>(
		eteq::Variable<double>::get(&data, shape, "B")));
	eteq::EVariable<double> c(eteq::VarptrT<double>(
		eteq::Variable<double>::get(&data, shape, "C")));
	eteq::EVariable<double> d(eteq::VarptrT<double>(
		eteq::Variable<double>::get(&data, shape, "D")));

	marsh::Maps attrs;
	teq::TensptrT u(eteq::Functor<double>::get(
		egen::NEG, {a}, std::move(attrs)));
	teq::TensptrT x(eteq::Functor<double>::get(
		egen::MUL, {u, b}, std::move(attrs)));
	teq::TensptrT y(eteq::Functor<double>::get(
		egen::DIV, {c, u}, std::move(attrs)));
	teq::TensptrT z(eteq::Functor<double>::get(
		egen::SUB, {y, x}, std::move(attrs)));
	teq::TensptrT target(eteq::Functor<double>::get(
		egen::POW, {z, d}, std::move(attrs)));

	eteq::ETensor targetens(target);
	eteq::ETensor ztens(z);
	eteq::ETensor ytens(y);
	eteq::ETensor xtens(x);
	eteq::ETensor utens(u);
	eteq::ETensor nothing;

	EXPECT_EQ(nullptr, nothing.calc<double>());

	targetens.calc<double>();

	// pow (targeted) = 512
	// `-- - (z) = 8
	// |   `-- / (y) = -1
	// |   |   `-- c = 3
	// |   |   `-- - (u) = -3
	// |   |       `-- a = 3
	// |   `-- * (x) = -9
	// |       `-- - (u)
	// |       |   `-- a = 3
	// |       `-- b = 3
	// `-- d = 3

	EXPECT_EQ(512, *targetens.data<double>());
	EXPECT_EQ(8, *ztens.data<double>());
	EXPECT_EQ(-1, *ytens.data<double>());
	EXPECT_EQ(-9, *xtens.data<double>());
	EXPECT_EQ(-3, *utens.data<double>());

	double nextdata = 2;
	a->assign(&nextdata, shape);
	// new state:
	// pow (targeted) = 512
	// `-- - (z) = 8
	// |   `-- / (y) = -1
	// |   |   `-- c = 3
	// |   |   `-- - (u) = -3
	// |   |       `-- a = 2
	// |   `-- * (x) = -9
	// |       `-- - (u)
	// |       |   `-- a = 2
	// |       `-- b = 3
	// `-- d = 3

	ztens.calc<double>(teq::TensSetT{y.get()});
	// expected state:
	// pow (targeted) = 512
	// `-- - (z) = 5
	// |   `-- / (y) = -1
	// |   |   `-- c = 3
	// |   |   `-- - (u) = -2
	// |   |       `-- a = 2
	// |   `-- * (x) = -6
	// |       `-- - (u)
	// |       |   `-- a = 2
	// |       `-- b = 3
	// `-- d = 3
	EXPECT_EQ(512, *targetens.data<double>());
	EXPECT_EQ(5, *ztens.data<double>());
	EXPECT_EQ(-1, *ytens.data<double>());
	EXPECT_EQ(-6, *xtens.data<double>());
	EXPECT_EQ(-2, *utens.data<double>());

	a->upversion(a->get_meta().state_version() + 1);
	// no change

	ztens.calc<double>();
	// expected state:
	// pow (targeted) = 512
	// `-- - (z) = 4.5
	// |   `-- / (y) = -1.5
	// |   |   `-- c = 3
	// |   |   `-- - (u) = -2
	// |   |       `-- a = 2
	// |   `-- * (x) = -6
	// |       `-- - (u)
	// |       |   `-- a = 2
	// |       `-- b = 3
	// `-- d = 3

	EXPECT_EQ(512, *targetens.data<double>());
	EXPECT_EQ(4.5, *ztens.data<double>());
	EXPECT_EQ(-1.5, *ytens.data<double>());
	EXPECT_EQ(-6, *xtens.data<double>());
	EXPECT_EQ(-2, *utens.data<double>());
}


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