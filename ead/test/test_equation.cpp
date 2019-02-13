
#ifndef DISABLE_EQUATION_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ead/generated/api.hpp"

#include "ead/session.hpp"
#include "ead/grader.hpp"
#include "ead/variable.hpp"


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

	ead::NodeptrT<int32_t> a = ead::make_variable<int32_t>(data.data(), ashape);
	ead::NodeptrT<int32_t> b = ead::make_variable<int32_t>(data2.data(), bshape);
	ead::NodeptrT<int32_t> c = ead::make_variable<int32_t>(data3.data(), cshape);

	auto d = age::matmul(a, b);
	auto e = age::matmul(c, d);
	auto f = age::matmul(age::transpose(d), age::transpose(c));
	auto dest = age::matmul(e, f);

	auto da = ead::derive(dest, a);
	auto db = ead::derive(dest, b);
	auto dc = ead::derive(dest, c);

	ead::Session<int32_t> session;
	session.track(dest);
	session.track(da);
	session.track(db);
	session.track(dc);
	session.update();

	auto ga = da->get_tensmap();
	auto gb = db->get_tensmap();
	auto gc = dc->get_tensmap();
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
