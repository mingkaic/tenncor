
#ifndef DISABLE_EQUATION_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/generated/api.hpp"

#include "llo/eval.hpp"
#include "llo/opt/derive.hpp"


TEST(EQUATION, MatmulComplex)
{
	std::vector<ade::DimT> alist = {3, 2};
	std::vector<ade::DimT> blist = {4, 3};
	std::vector<ade::DimT> clist = {2, 4};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(clist);

	std::vector<int32_t> data = {
		40, 1, 23,
		18, 50, 77,
	};
	std::vector<int32_t> data2 = {
		62, 31, 90, 68,
		68, 78, 55, 95,
		16, 99, 97, 77,
	};
	std::vector<int32_t> data3 = {
		29, 75,
		39, 67,
		37, 57,
		48, 42,
	};
	std::vector<int32_t> expect_ga = {
		-154880684, -127906804, -105914132,
		-206505460, -164002948, -131588540,
	};
	std::vector<int32_t> expect_gb = {
		-55352996, 20961008, -56860896, -29599272,
		-58614174, 26705056, -60728512, -32475964,
		-108808856, 47268840, -112469200, -59707840,
	};
	std::vector<int32_t> expect_gc = {
		152732, 4310652,
		-73239126, -139902552,
		-56297930, -101235528,
		-79671648, -172118240,
	};

	ade::TensptrT a(llo::Variable<int32_t>::get(data, ashape));
	ade::TensptrT b(llo::Variable<int32_t>::get(data2, bshape));
	ade::TensptrT c(llo::Variable<int32_t>::get(data3, cshape));

	auto d = age::fast_matmul(a, b);
	auto e = age::fast_matmul(c, d);
	auto f = age::fast_matmul(age::transpose(d), age::transpose(c));
	auto dest = age::fast_matmul(e, f);

	ade::TensT ds = llo::multi_derive(dest, {
		a.get(), b.get(), c.get()});

	auto da = ds[0];
	auto db = ds[1];
	auto dc = ds[2];

	llo::TensptrT<int32_t> ga = llo::eval<int32_t>(da);
	llo::TensptrT<int32_t> gb = llo::eval<int32_t>(db);
	llo::TensptrT<int32_t> gc = llo::eval<int32_t>(dc);
	{
		auto gotshape = ga->dimensions();
		ASSERT_ARREQ(alist, gotshape);
	}
	int32_t* gaptr = (int32_t*) ga->data();
	for (size_t i = 0, n = ashape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_ga[i], gaptr[i]);
	}
	{
		auto gotshape = gb->dimensions();
		ASSERT_ARREQ(blist, gotshape);
	}
	int32_t* gbptr = (int32_t*) gb->data();
	for (size_t i = 0, n = bshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gb[i], gbptr[i]);
	}
	{
		auto gotshape = gc->dimensions();
		ASSERT_ARREQ(clist, gotshape);
	}
	int32_t* gcptr = (int32_t*) gc->data();
	for (size_t i = 0, n = cshape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(expect_gc[i], gcptr[i]);
	}
}


#endif // DISABLE_EQUATION_TEST
