#ifndef DISABLE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/check.hpp"

#include "soil/external.hpp"
#include "soil/variable.hpp"
#include "soil/constant.hpp"
#include "soil/functor.hpp"

// #define DISABLE_EXTERNAL_TEST
#ifndef DISABLE_EXTERNAL_TEST


using namespace testutil;


class EXTERNAL : public fuzz_test {};


TEST_F(EXTERNAL, Transpose)
{
	// todo: tie to fuzz engine
	std::vector<double> adata = {1, 2, 3, 4, 5, 6};
	std::vector<double> expectout = {1, 3, 5, 2, 4, 6};
	Shape as({2, 3});
	Shape outshape({3, 2});

	Varptr a = Variable::get(as, DOUBLE);

	Nodeptr b = transpose(a);
	a.set_data<double>(adata);

	DataBucket bucket = evaluate(b);

	EXPECT_EQ(DTYPE::DOUBLE, bucket.type());

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}


TEST_F(EXTERNAL, TransposeDim01To2)
{
	// todo: tie to fuzz engine
	std::vector<double> adata = {
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12};
	std::vector<double> expectout = {
		1, 7, 2, 8, 3, 9,
		4, 10, 5, 11, 6, 12};
	Shape as({3, 2, 2});
	Shape outshape({2, 3, 2});

	Varptr a = Variable::get(as, DOUBLE);

	Nodeptr b = transpose(a, {{0, 2}, {2, 3}});
	a.set_data<double>(adata);

	DataBucket bucket = evaluate(b);

	EXPECT_EQ(DTYPE::DOUBLE, bucket.type());

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}


TEST_F(EXTERNAL, TransposeDim0To12)
{
	// todo: tie to fuzz engine
	std::vector<double> adata = {
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12};
	std::vector<double> expectout = {
		1, 4, 7, 10, 2, 5,
		8, 11, 3, 6, 9, 12};
	Shape as({3, 2, 2});
	Shape outshape({2, 2, 3});

	Varptr a = Variable::get(as, DOUBLE);

	Nodeptr b = transpose(a, {{0, 1}, {1, 3}});
	a.set_data<double>(adata);

	DataBucket bucket = evaluate(b);

	EXPECT_EQ(DTYPE::DOUBLE, bucket.type());

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}


TEST_F(EXTERNAL, OneLevelElem)
{
	// todo: tie to fuzz engine
	Shape outshape({3, 2});
	std::vector<double> adata = {0.5, 0.2, 0.3, 1, -0.7, -0.9};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> expectout = {
		1.5, -0.8, 1.3, 2, -1.7, -1.4};

	Varptr a = Variable::get(outshape, DOUBLE);
	Varptr b = Variable::get(outshape, DOUBLE);

	Nodeptr c = a + b;
	a.set_data<double>(adata);
	b.set_data<double>(bdata);

	DataBucket bucket = evaluate(c);

	EXPECT_EQ(DTYPE::DOUBLE, bucket.type());

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, MultiLevelElem)
{
	// todo: tie to fuzz engine
	Shape outshape({3, 2});
	std::vector<double> adata = {0.5, 0.2, 0.3, 1, -0.7, -0.9};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> cdata = {0.5, 0, -0.5, 1, -1, 0.25};
	std::vector<double> expectout = {
		1, 0.2, -0.2, 2, 0.3, -1.025};

	Nodeptr a = Constant::get(outshape, adata);
	Varptr b = Variable::get(outshape, DOUBLE);
	Varptr c = Variable::get(outshape, DOUBLE);

	Nodeptr d = a + b * c;
	b.set_data<double>(bdata);
	c.set_data<double>(cdata);

	DataBucket bucket = evaluate(d);

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, MultiLevelElemGrad)
{
	// todo: tie to fuzz engine
	Shape outshape({3, 2});
	std::vector<double> adata = {0.5, 0.2, 0.3, 1, -0.7, -0.9};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> cdata = {0.5, 0, -0.5, 1, -1, 0.25};
	std::vector<double> expectout = {
		1, -1, 1, 1, -1, -0.5};

	Nodeptr a = Constant::get(outshape, adata);
	Varptr b = Variable::get(outshape, DOUBLE);
	Varptr c = Variable::get(outshape, DOUBLE);

	Nodeptr d = a + b * c;
	b.set_data<double>(bdata);
	c.set_data<double>(cdata);

	Nodeptr gradc = d->gradient(c);
	DataBucket bucket = evaluate(gradc);

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, OneLevelMatrix)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape outshape({3, 3});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> expectout = {
		0.75, -0.75, 0.375,
		0.25, -0.25, -1.25,
		-1.75, 1.75, -0.25
	};

	Nodeptr a = Constant::get(ashape, adata);
	Varptr b = Variable::get(bshape, DOUBLE);

	Nodeptr c = matmul(a, b);
	b.set_data<double>(bdata);

	DataBucket bucket = evaluate(c);

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, MultiLevelMatrix)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape cshape({2, 3});
	Shape outshape({2, 3});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> cdata = {0.5, 0, -0.5, 1, -1, 0.25};
	std::vector<double> expectout = {
		0.375, -0.65625,
		1.5, -0.5625,
		-1.5, 1.6875
	};

	Nodeptr a = Constant::get(ashape, adata);
	Varptr b = Variable::get(bshape, DOUBLE);
	Varptr c = Variable::get(cshape, DOUBLE);

	Nodeptr d = matmul(a, matmul(b, c));
	b.set_data<double>(bdata);
	c.set_data<double>(cdata);

	DataBucket bucket = evaluate(d);

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshape.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, OneLevelMatrixGradLeft)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape outga({3, 3, 2, 3});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};

	std::vector<double> expectga = {
		1, -1, 1,    0, 0, 0, 0, 0, 0,
		1, -1, -0.5, 0, 0, 0, 0, 0, 0,

		0, 0, 0, 1, -1, 1,    0, 0, 0,
		0, 0, 0, 1, -1, -0.5, 0, 0, 0,

		0, 0, 0, 0, 0, 0, 1, -1, 1,
		0, 0, 0, 0, 0, 0, 1, -1, -0.5
	};

	Varptr a = Variable::get(ashape, DOUBLE);
	Varptr b = Variable::get(bshape, DOUBLE);

	Nodeptr c = matmul(a, b);
	a.set_data<double>(adata);
	b.set_data<double>(bdata);

	Nodeptr grada = c->gradient(a);
	DataBucket bucketa = evaluate(grada);
	Shape rawshape = bucketa.shape();
	std::vector<DimT> expectshape = outga.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucketa.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectga.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectga[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, OneLevelMatrixGradRight)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape outgb({3, 3, 3, 2});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};

	std::vector<double> expectgb = {
		0.5, 0,   0,   -0.75, 0,     0,     -0.75, 0,     0,
		0,   0.5, 0,   0,     -0.75, 0,     0,     -0.75, 0,
		0,   0,   0.5, 0,     0,     -0.75, 0,     0,     -0.75,

		0.25, 0,    0,    1, 0, 0, -1, 0,  0,
		0,    0.25, 0,    0, 1, 0, 0,  -1, 0,
		0,    0,    0.25, 0, 0, 1, 0,  0,  -1
	};

	Varptr a = Variable::get(ashape, DOUBLE);
	Varptr b = Variable::get(bshape, DOUBLE);

	Nodeptr c = matmul(a, b);
	a.set_data<double>(adata);
	b.set_data<double>(bdata);

	Nodeptr gradb = c->gradient(b);
	DataBucket bucketb = evaluate(gradb);
	Shape rawshape = bucketb.shape();
	std::vector<DimT> expectshape = outgb.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucketb.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectgb.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectgb[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, TwoLevelMatrixGrad)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape cshape({2, 3});
	Shape outshaped({2, 3, 2, 3});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> cdata = {0.5, 0, -0.5, 1, -1, 0.25};
	// c = matmul(a, b)
	// dc/dx = matmul(dc/da[shape:ac], da/dx[shape:xa])[shape:xc] +
	//		  matmul(dc/db[shape:bc], db/dx[shape:xb])[shape:xc]

	// h = matmul(b, c)
	// dh/dc[shape:hc]

	// g = matmul(a, h)
	// dg/dc = matmul(da/dc, dg/da) + matmul(dh/dc, dg/dh)
	//	   = matmul(dh/dc[shape:hx], dg/dh[shape:gh])[shape:gx]

	// f = matmul(g, z)
	// df/dc = matmul(dg/dc, df/dg) + matmul(dz/dc, df/dz)
	//	   = matmul(dg/dc[shape:gx], df/dg[shape:fg])[shape:fx]

	// dh/dc = [
	//	1,  0,  1,	0,
	//	0,  1,  0,	1,
	//	-1, 0,  -1,   0,
	//	0,  -1, 0,	-1,
	//	1,  0,  -0.5, 0,
	//	0,  1,  0,	-0.5
	// ]

	// dg/dh = [
	//	0.5,  0,	-0.75, 0,	 -0.75, 0,
	//	0,	0.5,  0,	 -0.75, 0,	 -0.75,
	//	0.25, 0,	1,	 0,	 -1,	0,
	//	0,	0.25, 0,	 1,	 0,	 -1,
	// ]

	std::vector<double> expectout = {
		0.75, 0, 0.25, 0, -1.75, 0,
		0, 0.75, 0, 0.25, 0, -1.75,
		-0.75, 0, -0.25, 0, 1.75, 0,
		0, -0.75, 0, -0.25, 0, 1.75,
		0.375, 0, -1.25, 0, -0.25, 0,
		0, 0.375, 0, -1.25, 0, -0.25
	};

	Nodeptr a = Constant::get(ashape, adata);
	Varptr b = Variable::get(bshape, DOUBLE);
	Varptr c = Variable::get(cshape, DOUBLE);

	Nodeptr d = matmul(a, matmul(b, c));
	b.set_data<double>(bdata);
	c.set_data<double>(cdata);

	Nodeptr gradc = d->gradient(c);
	DataBucket bucket = evaluate(gradc);
	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshaped.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectout.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectout[i], rawout[i]) <<
			"failed at index " << i;
	}
}

TEST_F(EXTERNAL, MultiLevelMatrixGrad)
{
	// todo: tie to fuzz engine
	Shape ashape({2, 3});
	Shape bshape({3, 2});
	Shape cshape({2, 3});
	Shape zshape({1, 2});
	Shape outshapef({1, 3, 2, 3});
	std::vector<double> adata = {0.5, 0.25, -0.75, 1, -0.75, -1};
	std::vector<double> bdata = {1, -1, 1, 1, -1, -0.5};
	std::vector<double> cdata = {0.5, 0, -0.5, 1, -1, 0.25};
	std::vector<double> zdata = {3, 4};
	// c = matmul(a, b)
	// dc/dx = matmul(dc/da[shape:ac], da/dx[shape:xa])[shape:xc] +
	//		  matmul(dc/db[shape:bc], db/dx[shape:xb])[shape:xc]

	// h = matmul(b, c)
	// dh/dc[shape:hc]

	// g = matmul(a, h)
	// dg/dc = matmul(da/dc, dg/da) + matmul(dh/dc, dg/dh)
	//	   = matmul(dh/dc[shape:hx], dg/dh[shape:gh])[shape:gx]

	// f = matmul(g, z)
	// df/dc = matmul(dg/dc, df/dg) + matmul(dz/dc, df/dz)
	//	   = matmul(dg/dc[shape:gx], df/dg[shape:fg])[shape:fx]

	// df/dg = [
	// 	3, 0, 0,
	// 	4, 0, 0,

	// 	0, 3, 0,
	// 	0, 4, 0,

	// 	0, 0, 3,
	// 	0, 0, 4
	// ]

	std::vector<double> expectoutf = {
		2.25, 0.75, -5.25,
		3, 1, -7,

		-2.25, -0.75, 5.25,
		-3, -1, 7,

		1.125, -3.75, -0.75,
		1.5, -5, -1
	};

	Nodeptr a = Constant::get(ashape, adata);
	Varptr b = Variable::get(bshape, DOUBLE);
	Varptr c = Variable::get(cshape, DOUBLE);
	Varptr z = Variable::get(zshape, zdata);

	Nodeptr f = matmul(matmul(a, matmul(b, c)), z);
	b.set_data<double>(bdata);
	c.set_data<double>(cdata);

	Nodeptr gradc = f->gradient(c);
	DataBucket bucket = evaluate(gradc);

	Shape rawshape = bucket.shape();
	std::vector<DimT> expectshape = outshapef.as_list();
	std::vector<DimT> slist = rawshape.as_list();
	std::vector<unsigned> expectlist(expectshape.begin(), expectshape.end());
	std::vector<unsigned> gotlist(slist.begin(), slist.end());
	EXPECT_ARREQ(expectlist, gotlist);

	std::vector<double> rawout = bucket.vectorize<double>();
	size_t nraw = rawout.size();
	ASSERT_EQ(expectoutf.size(), nraw);
	for (size_t i = 0; i < nraw; ++i)
	{
		EXPECT_DOUBLE_EQ(expectoutf[i], rawout[i]) <<
			"failed at index " << i;
	}
}


#endif /* DISABLE_EXTERNAL_TEST */


#endif /* DISABLE_MODULE_TESTS */
