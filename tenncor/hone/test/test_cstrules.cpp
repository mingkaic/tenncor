
#ifndef DISABLE_HONE_CSTRULES_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "tenncor/hone/hone.hpp"


TEST(CSTRULES, Typical)
{
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	teq::TensptrT a = eteq::make_constant<double>(data.data(), shape);
	teq::TensptrT b = eteq::make_constant<double>(data2.data(), shape);
	teq::TensptrT c = eteq::make_constant_scalar<double>(4, shape);

	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));
	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));
	teq::TensptrT everyone(eteq::make_functor(egen::ADD, {lhs, rhs}));

	std::ifstream rulefile("cfg/optimizations.json");
	auto optres = hone::optimize({everyone}, rulefile);
	ASSERT_EQ(1, optres.size());
	everyone = optres.front();

	EXPECT_GRAPHEQ(
		"(constant:[107\\44\\180\\90\\193\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n", everyone);
}


TEST(CSTRULES, StopAtVar)
{
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	teq::TensptrT a = eteq::make_variable<double>(data.data(), shape, "a");
	teq::TensptrT b = eteq::make_constant<double>(data2.data(), shape);
	teq::TensptrT c = eteq::make_constant_scalar<double>(4, shape);

	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));
	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));
	teq::TensptrT everyone(eteq::make_functor(egen::ADD, {lhs, rhs}));

	std::ifstream rulefile("cfg/optimizations.json");
	auto optres = hone::optimize({everyone}, rulefile);
	ASSERT_EQ(1, optres.size());
	everyone = optres.front();

	EXPECT_GRAPHEQ(
		"(ADD<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[26\\19\\78\\42\\65\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(ADD<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:a<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n", everyone);
}


TEST(CSTRULES, IdentityDependency)
{
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	teq::TensptrT a = eteq::make_constant<double>(data.data(), shape);
	teq::TensptrT b = eteq::make_constant<double>(data2.data(), shape);
	teq::TensptrT c = eteq::make_constant_scalar<double>(4, shape);

	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));
	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));
	teq::TensptrT identity(eteq::make_functor(egen::IDENTITY, {lhs, rhs}));

	std::ifstream rulefile("cfg/optimizations.json");
	auto optres = hone::optimize({identity}, rulefile);
	ASSERT_EQ(1, optres.size());
	identity = optres.front();

	EXPECT_GRAPHEQ(
		"(IDENTITY<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[26\\19\\78\\42\\65\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[81\\25\\102\\48\\128\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n", identity);
}


#endif // DISABLE_HONE_CSTRULES_TEST
