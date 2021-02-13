
#ifndef DISABLE_TENNCOR_API_TEST


#include "testutil/tutil.hpp"

#include "internal/utils/coord/coord.hpp"

#include "tenncor/tenncor.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


using UnaryDblF = std::function<double(double)>;

template <typename T>
using UnaryOpF = std::function<eteq::ETensor(eteq::ETensor&)>;

template <typename T>
using BinaryOpF = std::function<eteq::ETensor(eteq::ETensor&,eteq::ETensor&)>;

template <typename T>
using LhsBinaryOpF = std::function<eteq::ETensor(eteq::ETensor&,T&)>;

template <typename T>
using RhsBinaryOpF = std::function<eteq::ETensor(T&,eteq::ETensor&)>;

template <typename T>
using BinaryFwdF = std::function<T(T,T)>;

template <typename T>
using BinaryBwdF = std::function<T(T,T,T,T)>;

using MatVecT = std::vector<std::vector<int32_t>>;

static const int FREIVALD_N = 10;


static MatVecT create_2d (eteq::ETensor data,
	std::pair<teq::RankT,teq::RankT> dims = {0, 1})
{
	int32_t* ptr = (int32_t*) data->device().data();
	teq::DimT C = data->shape().at(dims.first);
	teq::DimT R = data->shape().at(dims.second);
	MatVecT res;

 	for (size_t y = 0; y < R; y++)
	{
		res.push_back(std::vector<signed>(C, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			res[y][x] = ptr[x + y * C];
		}
	}
	return res;
}


static bool freivald (MatVecT a, MatVecT b, MatVecT c)
{
	teq::RankT cdim = b.size();
	teq::RankT bdim = b[0].size();
	teq::RankT adim = a.size();
	// a has shape [cdim, adim]
	// b has shape [bdim, cdim]
	// c has shape [bdim, adim]
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	for (int i = 0; i < FREIVALD_N; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<int32_t> r(bdim);
		std::generate(r.begin(), r.end(),
			global::get_generator()->unif_intgen(0, 1));

		// p = matmul(a, matmul(b, r)) - matmul(c, r)
		std::vector<int32_t> br; // matmul(b, r)
		for (size_t y = 0; y < cdim; y++)
		{
			int32_t bri = 0;
			for (size_t x = 0; x < bdim; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<int32_t> cr; // matmul(c, r)
		for (size_t y = 0; y < adim; y++)
		{
			int32_t cri = 0;
			for (size_t x = 0; x < bdim; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<int32_t> p;
		for (size_t y = 0; y < adim; y++)
		{
			int32_t ari = 0;
			for (size_t x = 0, m = a[y].size(); x < m; x++)
			{
				ari += a[y][x] * br[x];
			}
			p.push_back(ari);
		}
		for (size_t j = 0; j < adim; j++)
		{
			p[j] -= cr[j];
		}

		// if p != 0 -> return false
		if (!std::all_of(p.begin(), p.end(),
			[](int32_t d) { return d == 0; }))
		{
			return false;
		}
	}
	return true;
}


static void unary_generic (UnaryOpF<double> op,
	std::function<void(eteq::ETensor,teq::Shape&,std::vector<double>&)> verify,
	std::function<void(double*,std::vector<double>&)> bwverify)
{
	eigen::Device device;
	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = op(src);

	if (auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get()))
	{
		eigen::Device(true).calc(*dtens,0);
		eigen::Device(true).calc(*dtens,0); // idempotency check
	}
	verify(dest, shape, data);

	teq::Evaluator eval;
	eteq::ETensorsT gsrc = tcr::derive(dest, {src});
	ASSERT_EQ(1, gsrc.size());
	auto gs = gsrc.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	eval.evaluate(device, {gs.get()});
	eval.evaluate(device, {gs.get()}); // idempotency check

	auto gotshape = gs->shape();
	ASSERT_ARREQ(shape, gotshape);
	double* goptr = (double*) gs->device().data();
	bwverify(goptr, data);

	eteq::DerivativeFuncs dfuncs;
	auto gsrc2 = teq::backprop(dest, {src}, dfuncs);
	ASSERT_EQ(1, gsrc2.size());
	auto gs2 = gsrc2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});

	auto gotshape2 = gs2->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr2 = (double*) gs2->device().data();
	bwverify(goptr2, data);
}


static void unar_elem (std::vector<double> data,
	teq::DimsT shape_list,
	UnaryOpF<double> op, UnaryDblF fwd, UnaryDblF bwd)
{
	teq::Evaluator eval;
	eigen::Device device;
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = op(src);
	eteq::ETensor uninit_dest = op(src);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eval.evaluate(device, {dest.get()});
	eval.evaluate(device, {dest.get()}); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* optr = (double*) dest->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
		}
	}

	eteq::ETensorsT gsrc = tcr::derive(uninit_dest, {src});
	ASSERT_EQ(1, gsrc.size());
	auto gs = gsrc.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	{
		auto gotshape = gs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	auto goptr = (double*) gs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
	}

	eteq::DerivativeFuncs dfuncs;
	auto gsrc2 = teq::backprop(uninit_dest, {src}, dfuncs);
	ASSERT_EQ(1, gsrc2.size());
	auto gs2 = tenncor().cast<double>(gsrc2.front());
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});

	auto gotshape2 = gs2->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr2 = (double*) gs2->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i]), goptr2[i]);
	}
}


static void unary_elementary (UnaryOpF<double> op,
	UnaryDblF fwd, UnaryDblF bwd)
{
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	unar_elem(data, slist, op, fwd, bwd);

	// matrix optimized operation
	teq::DimsT slist_2d = {2, 3};
	std::vector<double> data_2d = {
		59, 10, 28,
		10, 67, 62,
	};
	unar_elem(data_2d, slist_2d, op, fwd, bwd);
}


static void binar_elem (std::vector<double> data, std::vector<double> data2,
	teq::DimsT shape_list, BinaryOpF<double> op,
	LhsBinaryOpF<double> lhs_op, RhsBinaryOpF<double> rhs_op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd, double cst)
{
	eigen::Device device;
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	assert(data.size() == n);
	assert(data2.size() == n);

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor src2 = eteq::make_constant<double>(data2.data(), shape);
	eteq::ETensor dest = op(src, src2);
	eteq::ETensor clhs = lhs_op(src, cst);
	eteq::ETensor crhs = rhs_op(cst, src2);

	teq::Evaluator eval;
	eval.evaluate(device, {dest.get()});
	eval.evaluate(device, {dest.get()}); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	eval.evaluate(device, {clhs.get(), crhs.get()});
	eval.evaluate(device, {clhs.get(), crhs.get()}); // idempotency check
	{
		auto gotshape = clhs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	{
		auto gotshape = crhs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* lptr = (double*) clhs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i], cst), lptr[i]);
	}
	double* rptr = (double*) crhs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(cst, data2[i]), rptr[i]);
	}

	eteq::ETensor dest2 = op(src, src);
	eteq::ETensorsT gsame = tcr::derive(dest2, {src});
	ASSERT_EQ(1, gsame.size());
	auto gs = gsame.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	eval.evaluate(device, {gs.get()});
	eval.evaluate(device, {gs.get()}); // idempotency check
	{
		auto gotshape = gs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1., 1.), goptr[i]);
	}

	eteq::ETensorsT gleft = tcr::derive(dest, {src});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<double>(gl);
	eval.evaluate(device, {gl.get()});
	eval.evaluate(device, {gl.get()}); // idempotency check
	{
		auto gotshape = gl->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr2 = (double*) gl->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1., 0.), goptr2[i]);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {src2});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<double>(gr);
	eval.evaluate(device, {gr.get()});
	eval.evaluate(device, {gr.get()}); // idempotency check
	{
		auto gotshape = gr->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr3 = (double*) gr->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0., 1.), goptr3[i]);
	}

	eteq::DerivativeFuncs dfuncs;
	auto gsame2 = teq::backprop(dest2, {src}, dfuncs);
	ASSERT_EQ(1, gsame2.size());
	auto gs2 = gsame2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});
	{
		auto gotshape = gs2->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* gptr = (double*) gs2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1., 1.), gptr[i]);
		}
	}

	auto gleft2 = teq::backprop(dest, {src}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();
	ASSERT_NE(nullptr, gl2 );
	eval.evaluate(device, {gl2 .get()});
	{
		auto gotshape = gl2 ->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* gptr = (double*) gl2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1., 0.), gptr[i]);
		}
	}

	auto gright2 = teq::backprop(dest, {src2}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();
	ASSERT_NE(nullptr, gr2 );
	eval.evaluate(device, {gr2 .get()});
	{
		auto gotshape = gr2 ->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* gptr = (double*) gr2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0., 1.), gptr[i]);
		}
	}
}


static void binary_elementary (BinaryOpF<double> op,
	LhsBinaryOpF<double> lhs_op, RhsBinaryOpF<double> rhs_op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	// tensor operation
	teq::DimsT slist = {3, 2, 4};
	std::vector<double> data = {
		0.0919361505, 0.5135099474, 0.3147548326, 0.0281299379, 0.3705218798, 0.6808164860,
		0.1933972592, 0.2326945471, 0.4600163558, 0.1600801317, 0.9942654588, 0.8739832345,
		0.9664644529, 0.6152766955, 0.8795922916, 0.6384690466, 0.3922073677, 0.5979097486,
		0.0425608731, 0.1178122813, 0.1594330664, 0.0926580999, 0.9309809737, 0.2119471989
	};
	std::vector<double> data2 = {
		0.2547977589, 0.8808089905, 0.4323663340, 0.5710527217, 0.6207772267, 0.8574923091,
		0.2315629833, 0.8740258926, 0.9239905856, 0.0346148639, 0.3255387878, 0.7443564112,
		0.0930828560, 0.9324878301, 0.6552622891, 0.8305292319, 0.9515416240, 0.3653033185,
		0.0504231590, 0.8494357051, 0.0908431573, 0.1567913571, 0.1211327459, 0.5269402648
	};

	double cst = 0.7819955055;

	binar_elem(data, data2, slist, op, lhs_op, rhs_op, fwd, bwd, cst);

	// matrix optimized operation
	teq::DimsT slist_2d = {3, 2};
	std::vector<double> data_2d = {
		0.0919361505, 0.5135099474, 0.3147548326,
		0.0281299379, 0.3705218798, 0.6808164860,
	};
	std::vector<double> data2_2d = {
		0.2547977589, 0.8808089905, 0.4323663340,
		0.5710527217, 0.6207772267, 0.8574923091,
	};

	binar_elem(data_2d, data2_2d, slist_2d, op, lhs_op, rhs_op, fwd, bwd, cst);
}


static void binar_elem_int (std::vector<int32_t> data, std::vector<int32_t> data2,
	teq::DimsT shape_list, BinaryOpF<int32_t> op,
	LhsBinaryOpF<int32_t> lhs_op, RhsBinaryOpF<int32_t> rhs_op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd, int32_t cst)
{
	eigen::Device device;
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	eteq::ETensor src = eteq::make_constant<int32_t>(data.data(), shape);
	eteq::ETensor src2 = eteq::make_constant<int32_t>(data2.data(), shape);
	eteq::ETensor dest = op(src, src2);
	eteq::ETensor clhs = lhs_op(src, cst);
	eteq::ETensor crhs = rhs_op(cst, src2);

	teq::Evaluator eval;
	eval.evaluate(device, {dest.get()});
	eval.evaluate(device, {dest.get()}); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* optr = (int32_t*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	eval.evaluate(device, {clhs.get(), crhs.get()});
	eval.evaluate(device, {clhs.get(), crhs.get()}); // idempotency check
	{
		auto gotshape = clhs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	{
		auto gotshape = crhs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* lptr = (int32_t*) clhs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], cst), lptr[i]);
	}
	int32_t* rptr = (int32_t*) crhs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(cst, data2[i]), rptr[i]);
	}

	eteq::ETensor dest2 = op(src, src);
	eteq::ETensorsT gsame = tcr::derive(dest2, {src});
	ASSERT_EQ(1, gsame.size());
	auto gs = gsame.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<int32_t>(gs);
	eval.evaluate(device, {gs.get()});
	eval.evaluate(device, {gs.get()}); // idempotency check
	{
		auto gotshape = gs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr = (int32_t*) gs->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data[i], 1., 1.), goptr[i]);
	}

	eteq::ETensorsT gleft = tcr::derive(dest, {src});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<int32_t>(gl);
	eval.evaluate(device, {gl.get()});
	eval.evaluate(device, {gl.get()}); // idempotency check
	{
		auto gotshape = gl->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gl->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 1., 0.), goptr2[i]);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {src2});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<int32_t>(gr);
	eval.evaluate(device, {gr.get()});
	eval.evaluate(device, {gr.get()}); // idempotency check
	{
		auto gotshape = gr->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gr->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 0., 1.), goptr3[i]);
	}

	eteq::DerivativeFuncs dfuncs;
	auto gsame2 = teq::backprop(dest2, {src}, dfuncs);
	ASSERT_EQ(1, gsame2.size());
	auto gs2 = gsame2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});
	{
		auto gotshape = gs2->shape();
		ASSERT_ARREQ(shape, gotshape);
		int32_t* gptr = (int32_t*) gs2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1., 1.), gptr[i]);
		}
	}

	auto gleft2 = teq::backprop(dest, {src}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();
	ASSERT_NE(nullptr, gl2 );
	eval.evaluate(device, {gl2 .get()});
	{
		auto gotshape = gl2 ->shape();
		ASSERT_ARREQ(shape, gotshape);
		int32_t* gptr = (int32_t*) gl2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1., 0.), gptr[i]);
		}
	}

	auto gright2 = teq::backprop(dest, {src2}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();
	ASSERT_NE(nullptr, gr2 );
	eval.evaluate(device, {gr2 .get()});
	{
		auto gotshape = gr2 ->shape();
		ASSERT_ARREQ(shape, gotshape);
		int32_t* gptr = (int32_t*) gr2->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0., 1.), gptr[i]);
		}
	}
}


static void binary_elementary_int (BinaryOpF<int32_t> op,
	LhsBinaryOpF<int32_t> lhs_op, RhsBinaryOpF<int32_t> rhs_op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	// tensor operation
	teq::DimsT slist = {4, 3, 2};
	std::vector<int32_t> data = {
		1, 2, 3, 0, 1, 2, 2, 1, 1, 3, 3, 1,
		2, 2, 3, 0, 1, 3, 3, 1, 2, 0, 0, 2
	};
	std::vector<int32_t> data2 = {
		0, 0, 2, 1, 3, 3, 2, 2, 3, 1, 2, 3,
		1, 3, 1, 3, 1, 0, 2, 1, 2, 2, 0, 1
	};

	int32_t cst = 2;

	binar_elem_int(data, data2, slist, op, lhs_op, rhs_op, fwd, bwd, cst);

	// matrix optimized operation
	teq::DimsT slist_2d = {4, 2};
	std::vector<int32_t> data_2d = {
		1, 2, 3, 0,
		1, 2, 2, 1,
	};
	std::vector<int32_t> data2_2d = {
		0, 0, 2, 1,
		3, 3, 2, 2,
	};

	binar_elem_int(data_2d, data2_2d, slist_2d, op, lhs_op, rhs_op, fwd, bwd, cst);
}


static void nnary_elementary (std::vector<std::vector<double>> datas,
	teq::DimsT shape_list,
	std::function<double(size_t)> calc_expect,
	std::function<double(size_t,size_t)> calc_grad)
{
	eigen::Device device;
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	eteq::ETensorsT srcs;
	srcs.reserve(datas.size());
	for (auto& data : datas)
	{
		assert(data.size() == n);
		srcs.push_back(eteq::make_constant<double>(data.data(), shape));
	}
	auto dest = tenncor().sum(srcs);

	teq::Evaluator eval;
	eval.evaluate(device, {dest.get()});
	eval.evaluate(device, {dest.get()}); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		double expect = calc_expect(i);
		EXPECT_DOUBLE_EQ(expect, optr[i]);
	}

	eteq::ETensorsT gsrc = tcr::derive(dest, {srcs});
	size_t m = datas.size();
	ASSERT_EQ(datas.size(), gsrc.size());
	for (size_t i = 0; i < m; ++i)
	{
		auto gs = gsrc[i];
		ASSERT_NE(nullptr, gs);
		gs = tenncor().cast<double>(gs);
		teq::TensSetT targets = {gs.get()};
		eval.evaluate(device, targets);
		eval.evaluate(device, targets); // idempotency check
		{
			auto gotshape = gs->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr = (double*) gs->device().data();
		for (size_t j = 0; j < n; ++j)
		{
			double expect = calc_grad(i, j);
			EXPECT_DOUBLE_EQ(expect, goptr[j]);
		}
	}
}


TEST(API, Assign)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	eigen::Device device;
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

	eteq::EVariable target1 = eteq::make_variable<double>(data.data(), shape);
	eteq::EVariable target2 = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor src = eteq::make_constant<double>(data2.data(), shape);

	auto ass1 = tenncor().assign(target1, src);
	auto ass2 = tenncor().assign(target2, -src);

	eigen::Device(true).calc(*ass1,0);
	eigen::Device(true).calc(*ass1,0); // idempotency check
	{
		auto gotshape = target1->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass1->shape();
		ASSERT_ARREQ(shape, gotshape2);
	}
	double* optr = (double*) target1->device().data();
	double* aptr = (double*) ass1->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(data2[i], optr[i]);
		EXPECT_DOUBLE_EQ(data2[i], aptr[i]);
	}

	teq::Evaluator eval;
	eval.evaluate(device, {ass2.get()});
	eval.evaluate(device, {ass2.get()}); // idempotency check
	{
		auto gotshape = target2->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass2->shape();
		ASSERT_ARREQ(shape, gotshape2);
	}
	double* optr2 = (double*) target2->device().data();
	double* aptr2 = (double*) ass2->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(-data2[i], optr2[i]);
		EXPECT_DOUBLE_EQ(-data2[i], aptr2[i]);
	}

	std::string fatalmsg = "Unsupported op derivation ASSIGN";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(2).WillRepeatedly(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(tcr::derive(ass1, {src}), fatalmsg.c_str());
	EXPECT_FATAL(tcr::derive(ass2, {src}), fatalmsg.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(API, AssignHighToLowPrecision)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	eigen::Device device;
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<float> data = {
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

	eteq::EVariable target1 = eteq::make_variable<float>(data.data(), shape, "target1");
	eteq::EVariable target2 = eteq::make_variable<float>(data.data(), shape, "target2");
	eteq::ETensor src = eteq::make_constant<double>(data2.data(), shape);

	auto ass1 = tenncor().assign(target1, src);
	auto ass2 = tenncor().assign(target2, -src);

	EXPECT_GRAPHEQ(
		"(ASSIGN<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(variable:target1<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(CAST<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n",
		ass1);

	EXPECT_GRAPHEQ(
		"(ASSIGN<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(variable:target2<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(CAST<FLOAT>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_____`--(NEG<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:[22\\15\\74\\38\\61\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n",
		ass2);

	// Assignments must take type of target even if source has higher precision
	ASSERT_EQ(egen::FLOAT, ass1->get_meta().type_code());
	ASSERT_EQ(egen::FLOAT, ass2->get_meta().type_code());

	teq::Evaluator eval;
	eval.evaluate(device, {ass1.get()});
	eval.evaluate(device, {ass1.get()}); // idempotency check
	{
		auto gotshape = target1->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass1->shape();
		ASSERT_ARREQ(shape, gotshape2);
	}
	float* optr = (float*) target1->device().data();
	float* aptr = (float*) ass1->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(data2[i], optr[i]);
		EXPECT_DOUBLE_EQ(data2[i], aptr[i]);
	}

	eval.evaluate(device, {ass2.get()});
	eval.evaluate(device, {ass2.get()}); // idempotency check
	{
		auto gotshape = target2->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass2->shape();
		ASSERT_ARREQ(shape, gotshape2);
	}
	float* optr2 = (float*) target2->device().data();
	float* aptr2 = (float*) ass2->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(-data2[i], optr2[i]);
		EXPECT_DOUBLE_EQ(-data2[i], aptr2[i]);
	}

	std::string fatalmsg = "Unsupported op derivation ASSIGN";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(2).WillRepeatedly(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(tcr::derive(ass1, {src}), fatalmsg.c_str());
	EXPECT_FATAL(tcr::derive(ass2, {src}), fatalmsg.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(API, Identity) // todo: check for non-idempotency
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().identity(tenncor().abs(a)); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); });
}


TEST(API, Cast) // check for non-idempotency
{
	unary_elementary(
		[](eteq::ETensor& a)
		{
			auto out = tenncor().cast<double>(tenncor().cast<int32_t>(
				tenncor().cos(a) * 2.1 + 4.5) / int32_t(2));
			EXPECT_GRAPH_STRUCTEQ(
				"(CAST<DOUBLE>)\n"
				"_`--(DIV<INT32>)\n"
				"_____`--(CAST<INT32>)\n"
				"_____|___`--(ADD<DOUBLE>)\n"
				"_____|_______`--(MUL<DOUBLE>)\n"
				"_____|_______|___`--(COS<DOUBLE>)\n"
				"_____|_______|___|___`--(constant:[59\\10\\28\\10\\67\\...]<DOUBLE>)\n"
				"_____|_______|___`--(EXTEND<DOUBLE>)\n"
				"_____|_______|_______`--(constant:2.1<DOUBLE>)\n"
				"_____|_______`--(EXTEND<DOUBLE>)\n"
				"_____|___________`--(constant:4.5<DOUBLE>)\n"
				"_____`--(EXTEND<INT32>)\n"
				"_________`--(constant:2<INT32>)\n", out); // make sure we don't double cast
			return out;
		},
		[](double d) { return double(int32_t(std::cos(d) * 2.1 + 4.5) / 2); },
		[](double d) { return -std::sin(d) * 2.1 / 2; });
}


TEST(API, IdentityDependency)
{
	eigen::Device device;
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

	eteq::EVariable target = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;

	auto ass = tenncor().assign(target, tenncor().identity(b, {c}));

	teq::Evaluator eval;
	eval.evaluate(device, {ass.get()});
	{
		auto gotshape = target->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass->shape();
		ASSERT_ARREQ(shape, gotshape2);
		double* optr = (double*) target->device().data();
		double* aptr = (double*) ass->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(data2[i], optr[i]);
			EXPECT_DOUBLE_EQ(data2[i], aptr[i]);
		}
	}
	{
		auto gotshape = c->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* cptr = (double*) c->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_DOUBLE_EQ(data[i] + data2[i], cptr[i]);
		}
	}

	// EXPECT_FATAL(tcr::derive(ass, {a}), "Unknown op DEPEND");
}


// ensures depend's observee is executed only once
TEST(API, DependsRunOnce)
{
	eigen::Device device;
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

	eteq::EVariable target = eteq::make_variable_scalar<double>(0, shape);
	eteq::ETensor a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;

	// assign add is non-idempotent
	auto ass = tenncor().assign(target, tenncor().identity(b, {c}));

	teq::Evaluator eval;
	eval.evaluate(device, {ass.get()});
	{
		auto gotshape = target->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto gotshape2 = ass->shape();
		ASSERT_ARREQ(shape, gotshape2);
		double* optr = (double*) target->device().data();
		double* aptr = (double*) ass->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(data2[i], optr[i]);
			EXPECT_DOUBLE_EQ(data2[i], aptr[i]);
		}
	}
	{
		auto gotshape = c->shape();
		ASSERT_ARREQ(shape, gotshape);
		double* cptr = (double*) c->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_DOUBLE_EQ(data[i] + data2[i], cptr[i]);
		}
	}
}


TEST(API, Abs)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().abs(a); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); });
}


TEST(API, Neg)
{
	auto fwd = [](double d) { return -d; };
	auto bwd = [](double d) { return -1.; };
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().neg(a); },
		fwd, bwd);
	unary_elementary(
		[](eteq::ETensor& a) { return -a; },
		fwd, bwd);
}


TEST(API, Sin)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().sin(a); },
		[](double d) { return std::sin(d); },
		[](double d) { return std::cos(d); });
}


TEST(API, Cos)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().cos(a); },
		[](double d) { return std::cos(d); },
		[](double d) { return -std::sin(d); });
}


TEST(API, Tan)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().tan(a); },
		[](double d) { return std::tan(d); },
		[](double d) {
			double denom = std::cos(d);
			return 1. / denom / denom;
		});
}


TEST(API, Exp)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().exp(a); },
		[](double d) { return std::exp(d); },
		[](double d) { return std::exp(d); });
}


TEST(API, Log)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().log(a); },
		[](double d) { return std::log(d); },
		[](double d) { return 1. / d; });
}


TEST(API, Sqrt)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().sqrt(a); },
		[](double d) { return std::sqrt(d); },
		[](double d) { return 1. / (2 * std::sqrt(d)); });
}


TEST(API, Round)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().round(a); },
		[](double d) { return std::round(d); },
		[](double d) { return 1.; });
}


TEST(API, Sigmoid)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().sigmoid(a); },
		[](double d) { return 1 / (1 + std::exp(-d)); },
		[](double d)
		{
			double sig = 1 / (1 + std::exp(-d));
			return sig * (1 - sig);
		});
}


TEST(API, Tanh)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().tanh(a); },
		[](double d)
		{
			double e2d = std::exp(2 * d);
			return (e2d - 1) / (e2d + 1);
		},
		[](double d)
		{
			double e2d = std::exp(2 * d);
			double tanh = (e2d - 1) / (e2d + 1);
			return 1 - tanh * tanh;
		});
}


TEST(API, Square)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().square(a); },
		[](double d) { return d * d; },
		[](double d) { return 2 * d; });
}


TEST(API, Cube)
{
	unary_elementary(
		[](eteq::ETensor& a) { return tenncor().cube(a); },
		[](double d) { return d * d * d; },
		[](double d) { return 3 * d * d; });
}


TEST(API, Pow)
{
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().pow(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().pow(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().pow(a, b); },
		[](double a, double b) { return std::pow(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg * b * std::pow(a, b - 1) +
				rightg * std::pow(a, b) * std::log(a);
		});
}


TEST(API, Add)
{
	auto fwd = [](double a, double b) { return a + b; };
	auto bwd = [](double a, double b, double leftg, double rightg)
		{ return leftg + rightg; };
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().add(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().add(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().add(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a + b; },
		[](eteq::ETensor& a, double b)
		{ return a + b; },
		[](double a, eteq::ETensor& b)
		{ return a + b; },
		fwd, bwd);
}


TEST(API, Sub)
{
	auto fwd = [](double a, double b) { return a - b; };
	auto bwd = [](double a, double b, double leftg, double rightg)
		{ return leftg - rightg; };
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().sub(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().sub(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().sub(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a - b; },
		[](eteq::ETensor& a, double& b)
		{ return a - b; },
		[](double& a, eteq::ETensor& b)
		{ return a - b; },
		fwd, bwd);
}


TEST(API, Mul)
{
	auto fwd = [](double a, double b) { return a * b; };
	auto bwd = [](double a, double b, double leftg, double rightg)
		{
			return leftg * b + rightg * a;
		};
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().mul(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().mul(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().mul(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a * b; },
		[](eteq::ETensor& a, double& b)
		{ return a * b; },
		[](double& a, eteq::ETensor& b)
		{ return a * b; },
		fwd, bwd);
}


TEST(API, Div)
{
	auto fwd = [](double a, double b) { return a / b; };
	auto bwd = [](double a, double b, double leftg, double rightg)
		{
			return (leftg * b - rightg * a) / (b * b);
		};
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().div(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().div(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().div(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a / b; },
		[](eteq::ETensor& a, double& b)
		{ return a / b; },
		[](double& a, eteq::ETensor& b)
		{ return a / b; },
		fwd, bwd);
}


TEST(API, Min)
{
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().min(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().min(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().min(a, b); },
		[](double a, double b) { return std::min(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			if (a > b)
			{
				return rightg;
			}
			else if (b > a)
			{
				return leftg;
			}
			// else
			return leftg + rightg;
		});
}


TEST(API, Max)
{
	binary_elementary(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().max(a, b); },
		[](eteq::ETensor& a, double& b)
		{ return tenncor().max(a, b); },
		[](double& a, eteq::ETensor& b)
		{ return tenncor().max(a, b); },
		[](double a, double b) { return std::max(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			if (a > b)
			{
				return leftg;
			}
			else if (b > a)
			{
				return rightg;
			}
			// else
			return leftg + rightg;
		});
}


TEST(API, Select)
{
	// tensor operation
	teq::DimsT slist = {3, 2, 4};
	std::vector<double> cond = {
		0, 0, 1, 0, 1, 1,
		1, 0, 1, 0, 0, 0,
		1, 1, 1, 0, 1, 0,
		0, 0, 0, 0, 1, 0
	};
	std::vector<double> data = {
		0.0919361505, 0.5135099474, 0.3147548326, 0.0281299379, 0.3705218798, 0.6808164860,
		0.1933972592, 0.2326945471, 0.4600163558, 0.1600801317, 0.9942654588, 0.8739832345,
		0.9664644529, 0.6152766955, 0.8795922916, 0.6384690466, 0.3922073677, 0.5979097486,
		0.0425608731, 0.1178122813, 0.1594330664, 0.0926580999, 0.9309809737, 0.2119471989
	};
	std::vector<double> data2 = {
		0.2547977589, 0.8808089905, 0.4323663340, 0.5710527217, 0.6207772267, 0.8574923091,
		0.2315629833, 0.8740258926, 0.9239905856, 0.0346148639, 0.3255387878, 0.7443564112,
		0.0930828560, 0.9324878301, 0.6552622891, 0.8305292319, 0.9515416240, 0.3653033185,
		0.0504231590, 0.8494357051, 0.0908431573, 0.1567913571, 0.1211327459, 0.5269402648
	};

	// matrix optimized operation
	teq::DimsT slist_2d = {3, 2};
	std::vector<double> cond_2d = {
		1, 1, 0,
		0, 1, 0,
	};
	std::vector<double> data_2d = {
		0.0919361505, 0.5135099474, 0.3147548326,
		0.0281299379, 0.3705218798, 0.6808164860,
	};
	std::vector<double> data2_2d = {
		0.2547977589, 0.8808089905, 0.4323663340,
		0.5710527217, 0.6207772267, 0.8574923091,
	};

	auto trinar_elem = [](std::vector<double> cond,
		std::vector<double> data, std::vector<double> data2,
		teq::DimsT shape_list)
	{
		eigen::Device device;
		teq::Shape shape(shape_list);
		teq::NElemT n = shape.n_elems();

		assert(data.size() == n);
		assert(data2.size() == n);

		eteq::ETensor cond_src =
			eteq::make_constant<double>(cond.data(), shape);
		eteq::ETensor src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor dest =
			tenncor().if_then_else(cond_src, src, src2);

		auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
		ASSERT_NE(nullptr, dtens);
		eigen::Device(true).calc(*dtens,0);
		eigen::Device(true).calc(*dtens,0); // idempotency check
		{
			auto gotshape = dest->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* optr = (double*) dest->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			double expect = (bool) cond[i] ? data[i] : data2[i];
			EXPECT_DOUBLE_EQ(expect, optr[i]);
		}

		teq::Evaluator eval;
		eteq::ETensor dest2 = tenncor().if_then_else(cond_src, src, src);
		EXPECT_EQ(dest2.get(), src.get());

		eteq::ETensorsT gleft = tcr::derive(dest, {src});
		ASSERT_EQ(1, gleft.size());
		auto gl = gleft.front();
		ASSERT_NE(nullptr, gl);
		gl = tenncor().cast<double>(gl);
		teq::TensSetT ltargets = {gl.get()};
		eval.evaluate(device, ltargets);
		eval.evaluate(device, ltargets); // idempotency check
		{
			auto gotshape = gl->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr = (double*) gl->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(cond[i], goptr[i]);
		}

		eteq::ETensorsT gright = tcr::derive(dest, {src2});
		ASSERT_EQ(1, gright.size());
		auto gr = gright.front();
		ASSERT_NE(nullptr, gr);
		gr = tenncor().cast<double>(gr);
		teq::TensSetT rtargets = {gr.get()};
		eval.evaluate(device, rtargets);
		eval.evaluate(device, rtargets); // idempotency check
		{
			auto gotshape = gr->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr2 = (double*) gr->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ((0==cond[i]), goptr2[i]);
		}
	};

	trinar_elem(cond, data, data2, slist);
	trinar_elem(cond_2d, data_2d, data2_2d, slist_2d);
}


TEST(API, Slice)
{
	eigen::Device device;
	teq::DimsT slist = {3, 2, 4};
	std::vector<double> data = {
		0.0919361505, 0.5135099474, 0.3147548326, 0.0281299379, 0.3705218798, 0.6808164860,
		0.1933972592, 0.2326945471, 0.4600163558, 0.1600801317, 0.9942654588, 0.8739832345,
		0.9664644529, 0.6152766955, 0.8795922916, 0.6384690466, 0.3922073677, 0.5979097486,
		0.0425608731, 0.1178122813, 0.1594330664, 0.0926580999, 0.9309809737, 0.2119471989,
	};

	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);

	eteq::ETensor src =
		eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = tenncor().slice(src, {{0, 3}, {1, 1}});

	teq::Shape exshape({3, 1, 4});
	std::vector<double> exdata = {
		0.0281299379, 0.3705218798, 0.6808164860,
		0.1600801317, 0.9942654588, 0.8739832345,
		0.6384690466, 0.3922073677, 0.5979097486,
		0.0926580999, 0.9309809737, 0.2119471989,
	};
	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	auto gotshape = dest->shape();
	ASSERT_ARREQ(exshape, gotshape);
	double* optr = (double*) dest->device().data();
	std::vector<double> gotdata(optr, optr + gotshape.n_elems());
	EXPECT_VECEQ(exdata, gotdata);

	std::vector<double> exdata2 = {
		0, 0, 0, 1, 1, 1,
		0, 0, 0, 1, 1, 1,
		0, 0, 0, 1, 1, 1,
		0, 0, 0, 1, 1, 1,
	};
	teq::Evaluator eval;
	eteq::ETensorsT gsrcs = tcr::derive(dest, {src});
	ASSERT_EQ(1, gsrcs.size());
	auto gs = gsrcs.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	auto gotshape2 = gs->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr = (double*) gs->device().data();
	std::vector<double> gotdata2(goptr, goptr + gotshape2.n_elems());
	EXPECT_VECEQ(exdata2, gotdata2);
}


TEST(API, Eq)
{
	auto fwd = [](int32_t a, int32_t b) { return a == b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().eq(a, b); },
		[](eteq::ETensor& a, int32_t& b)
		{ return tenncor().eq(a, b); },
		[](int32_t& a, eteq::ETensor& b)
		{ return tenncor().eq(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a == b; },
		[](eteq::ETensor& a, int32_t& b)
		{ return a == b; },
		[](int32_t& a, eteq::ETensor& b)
		{ return a == b; },
		fwd, bwd);
}


TEST(API, Neq)
{
	auto fwd = [](int32_t a, int32_t b) { return a != b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().neq(a, b); },
		[](eteq::ETensor& a, int32_t& b)
		{ return tenncor().neq(a, b); },
		[](int32_t& a, eteq::ETensor& b)
		{ return tenncor().neq(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a != b; },
		[](eteq::ETensor& a, int32_t& b)
		{ return a != b; },
		[](int32_t& a, eteq::ETensor& b)
		{ return a != b; },
		fwd, bwd);
}


TEST(API, Lt)
{
	auto fwd = [](int32_t a, int32_t b) { return a < b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().lt(a, b); },
		[](eteq::ETensor& a, int32_t& b)
		{ return tenncor().lt(a, b); },
		[](int32_t& a, eteq::ETensor& b)
		{ return tenncor().lt(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a < b; },
		[](eteq::ETensor& a, int32_t& b)
		{ return a < b; },
		[](int32_t& a, eteq::ETensor& b)
		{ return a < b; },
		fwd, bwd);
}


TEST(API, Gt)
{
	auto fwd = [](int32_t a, int32_t b) { return a > b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return tenncor().gt(a, b); },
		[](eteq::ETensor& a, int32_t& b)
		{ return tenncor().gt(a, b); },
		[](int32_t& a, eteq::ETensor& b)
		{ return tenncor().gt(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor& a, eteq::ETensor& b)
		{ return a > b; },
		[](eteq::ETensor& a, int32_t& b)
		{ return a > b; },
		[](int32_t& a, eteq::ETensor& b)
		{ return a > b; },
		fwd, bwd);
}


TEST(API, NElems)
{
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().n_elems(src); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out->shape().n_elems());
			double got = *((double*) out->device().data());

			EXPECT_EQ(shape.n_elems(), got);
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(0, gout[i]);
			}
		});
}


TEST(API, NDims)
{
	teq::RankT dim = 2;
	unary_generic(
		[dim](eteq::ETensor& src) { return tenncor().n_dims(src, dim); },
		[dim](eteq::ETensor out, teq::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out->shape().n_elems());
			double got = *((double*) out->device().data());

			EXPECT_EQ(shape.at(dim), got);
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(0, gout[i]);
			}
		});
}


TEST(API, Argmax)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = tenncor().argmax(src);

	if (auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get()))
	{
		eigen::Device(true).calc(*dtens,0);
		eigen::Device(true).calc(*dtens,0); // idempotency check
	}
	teq::Shape oshape = dest->shape();
	teq::Shape exshape;
	EXPECT_ARREQ(exshape, oshape);
	double* ptr = (double*) dest->device().data();
	EXPECT_EQ(8, *ptr);

	std::string fatalmsg = "Unsupported op derivation ARGMAX";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(tcr::derive(dest, {src}), fatalmsg.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(API, Rsum)
{
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_sum(src); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			{
				ASSERT_EQ(1, n);
			}
			double got = *((double*) out->device().data());

			double expect = std::accumulate(data.begin(), data.end(), 0.);
			EXPECT_DOUBLE_EQ(expect, got);
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(1, gout[i]);
			}
		});
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_sum(src, 1, 1); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			teq::DimsT expect_list = shape.to_list();
			expect_list[1] = 1;
			teq::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			teq::CoordT coord;
			teq::DimT d = shape.at(1);
			double* got = (double*) out->device().data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = teq::coordinate(gotshape, i);
				double acc = 0;
				for (size_t j = 0; j < d; ++j)
				{
					coord[1] = j;
					acc += data[teq::index(shape, coord)];
				}
				EXPECT_DOUBLE_EQ(acc, got[i]);
			}
		},
		[](double* gout, std::vector<double>& og)
		{
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(1, gout[i]);
			}
		});
}


TEST(API, Rprod)
{
	eigen::Device device;
	teq::DimsT slist = {2, 2, 3};
	teq::Shape shape(slist);
	std::vector<int32_t> data = {
		2, 1,
		7, 3,

		6, 9,
		6, 8,

		9, 7,
		7, 2,
	};

	eteq::ETensor src = eteq::make_constant<int32_t>(data.data(), shape);
	eteq::ETensor dest = tenncor().reduce_prod(src);
	eteq::ETensor dest2 = tenncor().reduce_prod(src, 1, 1);

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	{
		size_t n = dest->shape().n_elems();
		{
			ASSERT_EQ(1, n);
		}
		int32_t got = *((int32_t*) dest->device().data());

		int32_t expect = std::accumulate(data.begin(), data.end(), 1, std::multiplies<int32_t>());
		EXPECT_EQ(expect, got);
	}

	auto dtens2 = static_cast<eteq::Functor<int32_t>*>(dest2.get());
	ASSERT_NE(nullptr, dtens2);
	eigen::Device(true).calc(*dtens2,0);
	eigen::Device(true).calc(*dtens2,0); // idempotency check
	{
		teq::DimsT expect_list = shape.to_list();
		expect_list[1] = 1;
		teq::Shape gotshape = dest2->shape();
		EXPECT_ARREQ(expect_list, gotshape);

		teq::CoordT coord;
		teq::DimT d = shape.at(1);
		int32_t* got = (int32_t*) dest2->device().data();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			coord = teq::coordinate(gotshape, i);
			int32_t acc = 1;
			for (size_t j = 0; j < d; ++j)
			{
				coord[1] = j;
				acc *= data[teq::index(shape, coord)];
			}
			EXPECT_EQ(acc, got[i]);
		}
	}

	teq::Evaluator eval;
	eteq::ETensorsT gsrcs = tcr::derive(dest, {src});
	eteq::ETensorsT gsrcs2 = tcr::derive(dest2, {src});
	ASSERT_EQ(1, gsrcs.size());
	ASSERT_EQ(1, gsrcs2.size());
	auto gsrc = gsrcs.front();
	auto gsrc2 = gsrcs2.front();
	ASSERT_NE(nullptr, gsrc);
	ASSERT_NE(nullptr, gsrc2);
	gsrc = tenncor().cast<int32_t>(gsrc);
	gsrc2 = tenncor().cast<int32_t>(gsrc2);
	eval.evaluate(device, {gsrc.get(), gsrc2.get()});
	eval.evaluate(device, {gsrc.get(), gsrc2.get()}); // idempotency check

	auto gotshape = gsrc->shape();
	ASSERT_ARREQ(shape, gotshape);
	int32_t* goptr = (int32_t*) gsrc->device().data();
	{
		size_t n = data.size();
		std::vector<int32_t> left(n, 1);
		std::vector<int32_t> right(n, 1);
		for (size_t i = 1; i < n; ++i)
		{
			left[i] = data[i - 1] * left[i - 1];
			right[n - i - 1] = data[n - i] * right[n - i];
		}
		for (size_t i = 0; i < n; ++i)
		{
			int32_t expect = left[i] * right[i];
			EXPECT_EQ(expect, goptr[i]);
		}
	}

	std::vector<int32_t> ex_grad = {
		7, 3,
		2, 1,

		6, 8,
		6, 9,

		7, 2,
		9, 7,
	};
	auto gotshape2 = gsrc2->shape();
	ASSERT_ARREQ(shape, gotshape2);
	int32_t* goptr2 = (int32_t*) gsrc2->device().data();
	{
		for (size_t i = 0, n = ex_grad.size(); i < n; ++i)
		{
			EXPECT_EQ(ex_grad[i], goptr2[i]);
		}
	}
}


TEST(API, Rmin)
{
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_min(src); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			ASSERT_EQ(1, n);
			double got = *((double*) out->device().data());

			double expect = *(std::min_element(data.begin(), data.end()));
			EXPECT_DOUBLE_EQ(expect, got);
		},
		[](double* gout, std::vector<double>& og)
		{
			double bigly = *(std::min_element(og.begin(), og.end()));
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				if (og[i] == bigly)
				{
					EXPECT_EQ(1, gout[i]);
				}
				else
				{
					EXPECT_EQ(0, gout[i]);
				}
			}
		});
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_min(src, 1, 1); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			teq::DimsT expect_list = shape.to_list();
			expect_list[1] = 1;
			teq::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			teq::CoordT coord;
			teq::DimT d = shape.at(1);
			double* got = (double*) out->device().data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = teq::coordinate(gotshape, i);
				double acc = data[teq::index(shape, coord)];
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					acc = std::min(acc, data[teq::index(shape, coord)]);
				}
				EXPECT_DOUBLE_EQ(acc, got[i]);
			}
		},
		[](double* gout, std::vector<double>& og)
		{
			teq::Shape inshape({2, 3, 4});
			teq::Shape outshape({2, 1, 4});
			teq::CoordT coord;
			teq::DimT d = 3;
			size_t m = og.size();
			size_t n = outshape.n_elems();
			std::vector<double> expect(m, 0);
			for (size_t i = 0; i < n; ++i)
			{
				coord = teq::coordinate(outshape, i);
				size_t min_idx = teq::index(inshape, coord);
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					size_t idx = teq::index(inshape, coord);
					if (og[min_idx] > og[idx])
					{
						min_idx = idx;
					}
				}
				expect[min_idx] = 1;
			}
			std::vector<double> got(gout, gout + m);
			EXPECT_VECEQ(expect, got);
		});
}


TEST(API, Rmax)
{
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_max(src); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			ASSERT_EQ(1, n);
			double got = *((double*) out->device().data());

			double expect = *(std::max_element(data.begin(), data.end()));
			EXPECT_DOUBLE_EQ(expect, got);
		},
		[](double* gout, std::vector<double>& og)
		{
			double bigly = *(std::max_element(og.begin(), og.end()));
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				if (og[i] == bigly)
				{
					EXPECT_EQ(1, gout[i]);
				}
				else
				{
					EXPECT_EQ(0, gout[i]);
				}
			}
		});
	unary_generic(
		[](eteq::ETensor& src) { return tenncor().reduce_max(src, 1, 1); },
		[](eteq::ETensor out, teq::Shape& shape, std::vector<double>& data)
		{
			teq::DimsT expect_list = shape.to_list();
			expect_list[1] = 1;
			teq::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			teq::CoordT coord;
			teq::DimT d = shape.at(1);
			double* got = (double*) out->device().data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = teq::coordinate(gotshape, i);
				double acc = data[teq::index(shape, coord)];
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					acc = std::max(acc, data[teq::index(shape, coord)]);
				}
				EXPECT_DOUBLE_EQ(acc, got[i]);
			}
		},
		[](double* gout, std::vector<double>& og)
		{
			teq::Shape inshape({2, 3, 4});
			teq::Shape outshape({2, 1, 4});
			teq::CoordT coord;
			teq::DimT d = 3;
			size_t m = og.size();
			size_t n = outshape.n_elems();
			std::vector<double> expect(m, 0);
			for (size_t i = 0; i < n; ++i)
			{
				coord = teq::coordinate(outshape, i);
				size_t max_idx = teq::index(inshape, coord);
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					size_t idx = teq::index(inshape, coord);
					if (og[max_idx] < og[idx])
					{
						max_idx = idx;
					}
				}
				expect[max_idx] = 1;
			}
			std::vector<double> got(gout, gout + m);
			EXPECT_VECEQ(expect, got);
		});
}


TEST(API, Permute)
{
	eigen::Device device;
	teq::DimsT slist = {4, 3, 2};
	teq::RanksT pidx = {2, 0, 1};
	teq::Shape shape(slist);
	teq::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		70, 36, 93, 50, 59, 98, 39, 5, 54, 84, 100, 94,
		75, 64, 30, 17, 90, 79, 21, 54, 6, 7, 69, 53
	};

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = tenncor().permute(src, pidx);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	size_t n = dest->shape().n_elems();
	ASSERT_EQ(nelem, n);
	double* got = (double*) dest->device().data();
	teq::CoordT coord, temp;
	for (size_t i = 0; i < n; ++i)
	{
		coord = temp = teq::coordinate(shape, i);
		for (int32_t j = 0, n = slist.size(); j < n; ++j)
		{
			coord[j] = temp[pidx[j]];
		}

		EXPECT_EQ(data[i], got[teq::index(dest->shape(), coord)]);
	}

	teq::Evaluator eval;
	eteq::ETensorsT gsrc = tcr::derive(dest, {src});
	ASSERT_EQ(1, gsrc.size());
	auto gs = gsrc.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	teq::TensSetT tset = {gs.get()};
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	{
		auto gotshape = gs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gs->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}

	eteq::DerivativeFuncs dfuncs;
	auto gsrc2 = teq::backprop(dest, {src}, dfuncs);
	ASSERT_EQ(1, gsrc2.size());
	auto gs2 = gsrc2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});

	auto gotshape2 = gs2->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr2 = (double*) gs2->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr2[i]);
	}
}


TEST(API, Extend)
{
	eigen::Device device;
	teq::DimsT slist = {2, 5};
	teq::DimsT ext = {1, 3};
	teq::Shape shape(slist);
	teq::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		51, 42, 9, 43, 37, 36, 65, 95, 10, 33
	};

	eteq::ETensor src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor dest = tenncor().extend(src, slist.size(), ext);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	size_t ext_nelem = teq::Shape(ext).n_elems();
	auto extshape = dest->shape();
	teq::Shape expect_shape({2, 5, 1, 3});
	EXPECT_ARREQ(expect_shape, extshape);
	size_t n = extshape.n_elems();
	ASSERT_EQ(nelem * ext_nelem, n);
	double* got = (double*) dest->device().data();
	for (size_t i = 0; i < nelem; ++i)
	{
		for (size_t j = 0; j < ext_nelem; ++j)
		{
			EXPECT_EQ(data[i], got[i + j * nelem]);
		}
	}

	teq::Evaluator eval;
	eteq::ETensorsT gsrc = tcr::derive(dest, {src});
	ASSERT_EQ(1, gsrc.size());
	auto gs = gsrc.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<double>(gs);
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	{
		auto gotshape = gs->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gs->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(ext_nelem, goptr[i]);
	}

	eteq::DerivativeFuncs dfuncs;
	auto gsrc2 = teq::backprop(dest, {src}, dfuncs);
	ASSERT_EQ(1, gsrc2.size());
	auto gs2 = gsrc2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});

	auto gotshape2 = gs2->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr2 = (double*) gs2->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(ext_nelem, goptr2[i]);
	}
}


TEST(API, Matmul)
{
	eigen::Device device;
	teq::DimsT alist = {3, 2};
	teq::DimsT blist = {4, 3};
	teq::DimsT sqrlist = {3, 3};
	teq::Shape ashape(alist);
	teq::Shape bshape(blist);
	teq::Shape cshape(sqrlist);

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
		29, 75, 39,
		67, 37, 57,
		48, 42, 56,
	};
	std::vector<int32_t> expect_ga = {
		62+31+90+68, 68+78+55+95, 16+99+97+77,
		62+31+90+68, 68+78+55+95, 16+99+97+77,
	};
	std::vector<int32_t> expect_gb = {
		40+18, 40+18, 40+18, 40+18,
		50+1, 50+1, 50+1, 50+1,
		23+77, 23+77, 23+77, 23+77,
	};

	eteq::ETensor a = eteq::make_constant<int32_t>(data.data(), ashape);
	eteq::ETensor b = eteq::make_constant<int32_t>(data2.data(), bshape);
	eteq::ETensor dest = tenncor().matmul(a, b);

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	teq::Shape gotshape = dest->shape();
	EXPECT_EQ(4, gotshape.at(0));
	EXPECT_EQ(2, gotshape.at(1));
	int32_t* optr = (int32_t*) dest->device().data();
	ASSERT_NE(nullptr, optr);

	MatVecT dda = create_2d(a);
	MatVecT ddb = create_2d(b);
	MatVecT ddc = create_2d(dest);
	EXPECT_TRUE(freivald(dda, ddb, ddc));

	teq::Evaluator eval;
	eteq::ETensor c = eteq::make_constant<int32_t>(data3.data(), cshape);
	eteq::ETensor dest2 = tenncor().matmul(c, c);
	eteq::ETensorsT gsame = tcr::derive(dest2, {c});
	ASSERT_EQ(1, gsame.size());
	auto gs = gsame.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<int32_t>(gs);
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	teq::Shape gcshape = gs->shape();
	ASSERT_ARREQ(cshape, gcshape);

	eteq::ETensorsT gleft = tcr::derive(dest, {a});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<int32_t>(gl);
	teq::TensSetT ltargets = {gl.get()};
	eval.evaluate(device, ltargets);
	eval.evaluate(device, ltargets); // idempotency check
	teq::Shape gashape = gl->shape();
	{
		ASSERT_ARREQ(ashape, gashape);
		int32_t* ga = (int32_t*) gl->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {b});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<int32_t>(gr);
	teq::TensSetT rtargets = {gr.get()};
	eval.evaluate(device, rtargets);
	eval.evaluate(device, rtargets); // idempotency check
	teq::Shape gbshape = gr->shape();
	{
		ASSERT_ARREQ(bshape, gbshape);
		int32_t* gb = (int32_t*) gr->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}

	eteq::DerivativeFuncs dfuncs;
	teq::TensptrsT gsame2 = teq::backprop(dest2, {c}, dfuncs);
	ASSERT_EQ(1, gsame2.size());
	auto gs2 = gsame2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});
	teq::Shape gcshape2 = gs2->shape();
	ASSERT_ARREQ(cshape, gcshape2);

	teq::TensptrsT gleft2 = teq::backprop(dest, {a}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();
	ASSERT_NE(nullptr, gl2);
	eval.evaluate(device, {gl2.get()});
	teq::Shape gashape2 = gl2->shape();
	{
		ASSERT_ARREQ(ashape, gashape2);
		int32_t* ga = (int32_t*) gl2->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	teq::TensptrsT gright2 = teq::backprop(dest, {b}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();
	ASSERT_NE(nullptr, gr2);
	eval.evaluate(device, {gr2.get()});
	teq::Shape gbshape2 = gr2->shape();
	{
		ASSERT_ARREQ(bshape, gbshape2);
		int32_t* gb = (int32_t*) gr2->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


TEST(API, Contract)
{
	eigen::Device device;
	teq::DimsT alist = {3, 1, 2};
	teq::DimsT blist = {4, 1, 3};
	teq::DimsT sqrlist = {3, 2};
	teq::Shape ashape(alist);
	teq::Shape bshape(blist);
	teq::Shape cshape(sqrlist);

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
		29, 75, 39,
		67, 37, 57,
	};
	std::vector<int32_t> expect_ga = {
		62+31+90+68, 68+78+55+95, 16+99+97+77,
		62+31+90+68, 68+78+55+95, 16+99+97+77,
	};
	std::vector<int32_t> expect_gb = {
		40+18, 40+18, 40+18, 40+18,
		50+1, 50+1, 50+1, 50+1,
		23+77, 23+77, 23+77, 23+77,
	};

	eteq::ETensor a = eteq::make_constant<int32_t>(data.data(), ashape);
	eteq::ETensor b = eteq::make_constant<int32_t>(data2.data(), bshape);
	eteq::ETensor dest = tenncor().contract(a, b, {{0, 2}});

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	teq::Shape gotshape = dest->shape();
	EXPECT_EQ(4, gotshape.at(0));
	EXPECT_EQ(1, gotshape.at(1));
	EXPECT_EQ(1, gotshape.at(2));
	EXPECT_EQ(2, gotshape.at(3));
	int32_t* optr = (int32_t*) dest->device().data();
	ASSERT_NE(nullptr, optr);

	MatVecT dda = create_2d(a, {0, 2});
	MatVecT ddb = create_2d(b, {0, 2});
	MatVecT ddc = create_2d(dest, {0, 3});
	EXPECT_TRUE(freivald(dda, ddb, ddc));

	teq::Evaluator eval;
	eteq::ETensor c = eteq::make_constant<int32_t>(data3.data(), cshape);
	eteq::ETensor dest2 = tenncor().contract(c, c, {{0, 0}});
	eteq::ETensorsT gsame = tcr::derive(dest2, {c});
	ASSERT_EQ(1, gsame.size());
	auto gs = gsame.front();
	ASSERT_NE(nullptr, gs);
	gs = tenncor().cast<int32_t>(gs);
	teq::TensSetT targets = {gs.get()};
	eval.evaluate(device, targets);
	eval.evaluate(device, targets); // idempotency check
	teq::Shape gcshape = gs->shape();
	ASSERT_ARREQ(cshape, gcshape);

	eteq::ETensorsT gleft = tcr::derive(dest, {a});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<int32_t>(gl);
	teq::TensSetT ltargets = {gl.get()};
	eval.evaluate(device, ltargets);
	eval.evaluate(device, ltargets); // idempotency check
	teq::Shape gashape = gl->shape();
	{
		ASSERT_ARREQ(ashape, gashape);
		int32_t* ga = (int32_t*) gl->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {b});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<int32_t>(gr);
	teq::TensSetT rtargets = {gr.get()};
	eval.evaluate(device, rtargets);
	eval.evaluate(device, rtargets); // idempotency check
	teq::Shape gbshape = gr->shape();
	{
		ASSERT_ARREQ(bshape, gbshape);
		int32_t* gb = (int32_t*) gr->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}

	eteq::DerivativeFuncs dfuncs;
	teq::TensptrsT gsame2 = teq::backprop(dest2, {c}, dfuncs);
	ASSERT_EQ(1, gsame2.size());
	auto gs2 = gsame2.front();
	ASSERT_NE(nullptr, gs2);
	eval.evaluate(device, {gs2.get()});
	teq::Shape gcshape2 = gs2->shape();
	ASSERT_ARREQ(cshape, gcshape2);

	teq::TensptrsT gleft2 = teq::backprop(dest, {a}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();
	ASSERT_NE(nullptr, gl2);
	eval.evaluate(device, {gl2.get()});
	teq::Shape gashape2 = gl2->shape();
	{
		ASSERT_ARREQ(ashape, gashape2);
		int32_t* ga = (int32_t*) gl2->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	teq::TensptrsT gright2 = teq::backprop(dest, {b}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();
	ASSERT_NE(nullptr, gr2);
	eval.evaluate(device, {gr2.get()});
	teq::Shape gbshape2 = gr2->shape();
	{
		ASSERT_ARREQ(bshape, gbshape2);
		int32_t* gb = (int32_t*) gr2->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


static void test_rand_unif (teq::DimsT shape_list)
{
	eigen::Device device;
	double hi = 3.2234;
	double lo = 0.2547977589;
	teq::Shape shape(shape_list);

	eteq::ETensor src = eteq::make_constant_scalar<double>(lo, shape);
	eteq::ETensor src2 = eteq::make_constant_scalar<double>(hi, shape);
	eteq::ETensor dest = tenncor().random.rand_unif(src, src2);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) dest->device().data();
	size_t nelems = dest->shape().n_elems();
	for (size_t i = 0; i < nelems; ++i)
	{
		EXPECT_LT(lo, optr[i]);
		EXPECT_GT(hi, optr[i]);
	}

	teq::Evaluator eval;
	eteq::ETensorsT gleft = tcr::derive(dest, {src});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<double>(gl);
	teq::TensSetT ltargets = {gl.get()};
	eval.evaluate(device, ltargets);
	eval.evaluate(device, ltargets); // idempotency check
	{
		auto gotshape = gl->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr2 = (double*) gl->device().data();
	EXPECT_DOUBLE_EQ(0, goptr2[0]);

	eteq::ETensorsT gright = tcr::derive(dest, {src});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<int32_t>(gr);
	teq::TensSetT rtargets = {gr.get()};
	eval.evaluate(device, rtargets);
	eval.evaluate(device, rtargets); // idempotency check
	{
		auto gotshape = gr->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr3 = (double*) gr->device().data();
	EXPECT_DOUBLE_EQ(0, goptr3[0]);
}


TEST(API, RandUniform)
{
	// tensor operation
	teq::DimsT slist = {31, 21, 14};
	test_rand_unif(slist);

	// matrix optimized operation
	teq::DimsT slist_2d = {31, 14};
	test_rand_unif(slist_2d);
}


TEST(API, Convolution)
{
	eigen::Device device;
	teq::DimsT alist = {2, 4, 3, 3};
	teq::DimsT blist = {1, 2, 2, 1};
	teq::Shape shape(alist);
	teq::Shape kshape(blist);
	teq::DimsT expectslist = {
		2, 3, 2, 3, 1, 1, 1, 1,
	};

	std::vector<double> data = {
		37, 93, 33, 47, 87, 39, 69, 10,
		74, 67, 32, 4, 99, 89, 85, 64,
		49, 61, 27, 89, 100, 41, 52, 66,

		71, 54, 7, 90, 8, 89, 20, 53,
		59, 32, 66, 55, 71, 37, 7, 98,
		48, 66, 58, 77, 61, 20, 48, 31,

		71, 20, 30, 26, 48, 3, 15, 78,
		70, 70, 58, 11, 58, 82, 57, 56,
		39, 88, 58, 63, 35, 69, 35, 62
	};
	std::vector<double> data2 = {
		2,3,
		2,4,
	};
	std::vector<double> expect_out = {
		449, 477, 787, 575, 919, 542,
		450, 624, 815, 617, 861, 716,

		545, 662, 454, 705, 246, 803,
		644, 669, 705, 455, 477, 532,

		604, 302, 552, 411, 485, 628,
		624, 601, 546, 670, 497, 718
	};
	std::vector<double> expect_ga = {
		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4,

		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4,

		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4
	};
	std::vector<double> expect_gb = {
		1887, 1781,
		2083, 2021,
	};

	eteq::ETensor img = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor kernel = eteq::make_constant<double>(data2.data(), kshape);
	teq::RanksT dims(teq::rank_cap);
	std::iota(dims.begin(), dims.end(), 0);
	eteq::ETensor dest = tenncor().convolution(img, kernel, dims);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(expectslist, gotshape);

		double* optr = (double*) dest->device().data();
		ASSERT_NE(nullptr, optr);
		std::vector<double> outdata(optr, optr + gotshape.n_elems());
		ASSERT_VECEQ(expect_out, outdata);
	}

	teq::Evaluator eval;
	eteq::ETensorsT gleft = tcr::derive(dest, {img});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<double>(gl);
	teq::TensSetT ltargets = {gl.get()};
	eval.evaluate(device, ltargets);
	eval.evaluate(device, ltargets); // idempotency check
	{
		auto gashape = gl->shape();
		ASSERT_ARREQ(shape, gashape);
		double* ga = (double*) gl->device().data();
		std::vector<double> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {kernel});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<double>(gr);
	teq::TensSetT rtargets = {gr.get()};
	eval.evaluate(device, rtargets);
	eval.evaluate(device, rtargets); // idempotency check
	{
		auto gbshape = gr->shape();
		ASSERT_ARREQ(kshape, gbshape);
		double* gb = (double*) gr->device().data();
		std::vector<double> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}

	eteq::DerivativeFuncs dfuncs;
	teq::TensptrsT gleft2 = teq::backprop(dest, {img}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();

	ASSERT_NE(nullptr, gl2);
	eval.evaluate(device, {gl2.get()});
	teq::Shape gashape2 = gl2->shape();
	{
		ASSERT_ARREQ(shape, gashape2);
		double* ga = (double*) gl2->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<double> ga_data(ga, ga + gashape2.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	teq::TensptrsT gright2 = teq::backprop(dest, {kernel}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();

	ASSERT_NE(nullptr, gr2);
	eval.evaluate(device, {gr2.get()});
	teq::Shape gbshape2 = gr2->shape();
	{
		ASSERT_ARREQ(kshape, gbshape2);
		double* gb = (double*) gr2->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<double> gb_data(gb, gb + gbshape2.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


TEST(API, ConvolutionWithDimensions)
{
	eigen::Device device;
	teq::DimsT alist = {2, 4, 3, 3};
	teq::DimsT blist = {2, 2};
	teq::Shape shape(alist);
	teq::Shape kshape(blist);
	teq::DimsT expectslist = {
		2, 3, 2, 3, 1, 1, 1, 1,
	};

	std::vector<double> data = {
		37, 93, 33, 47, 87, 39, 69, 10,
		74, 67, 32, 4, 99, 89, 85, 64,
		49, 61, 27, 89, 100, 41, 52, 66,

		71, 54, 7, 90, 8, 89, 20, 53,
		59, 32, 66, 55, 71, 37, 7, 98,
		48, 66, 58, 77, 61, 20, 48, 31,

		71, 20, 30, 26, 48, 3, 15, 78,
		70, 70, 58, 11, 58, 82, 57, 56,
		39, 88, 58, 63, 35, 69, 35, 62
	};
	std::vector<double> data2 = {
		2,3,
		2,4,
	};
	std::vector<double> expect_out = {
		449, 477, 787, 575, 919, 542,
		450, 624, 815, 617, 861, 716,

		545, 662, 454, 705, 246, 803,
		644, 669, 705, 455, 477, 532,

		604, 302, 552, 411, 485, 628,
		624, 601, 546, 670, 497, 718
	};
	std::vector<double> expect_ga = {
		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4,

		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4,

		2, 2, 5, 5, 5, 5, 3, 3,
		4, 4, 11, 11, 11, 11, 7, 7,
		2, 2, 6, 6, 6, 6, 4, 4
	};
	std::vector<double> expect_gb = {
		1887, 1781,
		2083, 2021,
	};

	eteq::ETensor img = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor kernel = eteq::make_constant<double>(data2.data(), kshape);
	teq::RanksT dims = {1, 2};
	eteq::ETensor dest = tenncor().convolution(img, kernel, dims);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::Device(true).calc(*dtens,0);
	eigen::Device(true).calc(*dtens,0); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(expectslist, gotshape);

		double* optr = (double*) dest->device().data();
		ASSERT_NE(nullptr, optr);
		std::vector<double> outdata(optr, optr + gotshape.n_elems());
		ASSERT_VECEQ(expect_out, outdata);
	}

	teq::Evaluator eval;
	eteq::ETensorsT gleft = tcr::derive(dest, {img});
	ASSERT_EQ(1, gleft.size());
	auto gl = gleft.front();
	ASSERT_NE(nullptr, gl);
	gl = tenncor().cast<double>(gl);
	teq::TensSetT ltargets = {gl.get()};
	eval.evaluate(device, ltargets);
	eval.evaluate(device, ltargets); // idempotency check
	{
		auto gashape = gl->shape();
		ASSERT_ARREQ(shape, gashape);
		double* ga = (double*) gl->device().data();
		std::vector<double> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT gright = tcr::derive(dest, {kernel});
	ASSERT_EQ(1, gright.size());
	auto gr = gright.front();
	ASSERT_NE(nullptr, gr);
	gr = tenncor().cast<double>(gr);
	teq::TensSetT rtargets = {gr.get()};
	eval.evaluate(device, rtargets);
	eval.evaluate(device, rtargets); // idempotency check
	{
		auto gbshape = gr->shape();
		ASSERT_ARREQ(kshape, gbshape);
		double* gb = (double*) gr->device().data();
		std::vector<double> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}

	eteq::DerivativeFuncs dfuncs;
	teq::TensptrsT gleft2 = teq::backprop(dest, {img}, dfuncs);
	ASSERT_EQ(1, gleft2.size());
	auto gl2 = gleft2.front();

	ASSERT_NE(nullptr, gl2);
	eval.evaluate(device, {gl2.get()});
	teq::Shape gashape2 = gl2->shape();
	{
		ASSERT_ARREQ(shape, gashape2);
		double* ga = (double*) gl2->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<double> ga_data(ga, ga + gashape2.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	teq::TensptrsT gright2 = teq::backprop(dest, {kernel}, dfuncs);
	ASSERT_EQ(1, gright2.size());
	auto gr2 = gright2.front();

	ASSERT_NE(nullptr, gr2);
	eval.evaluate(device, {gr2.get()});
	teq::Shape gbshape2 = gr2->shape();
	{
		ASSERT_ARREQ(kshape, gbshape2);
		double* gb = (double*) gr2->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<double> gb_data(gb, gb + gbshape2.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


TEST(API, GroupSum)
{
	// tensor operation
	teq::DimsT slist = {3, 2, 4};
	std::vector<double> data = {
		0.0919361505, 0.5135099474, 0.3147548326, 0.0281299379, 0.3705218798, 0.6808164860,
		0.1933972592, 0.2326945471, 0.4600163558, 0.1600801317, 0.9942654588, 0.8739832345,
		0.9664644529, 0.6152766955, 0.8795922916, 0.6384690466, 0.3922073677, 0.5979097486,
		0.0425608731, 0.1178122813, 0.1594330664, 0.0926580999, 0.9309809737, 0.2119471989,
	};
	std::vector<double> data2 = {
		0.2547977589, 0.8808089905, 0.4323663340, 0.5710527217, 0.6207772267, 0.8574923091,
		0.2315629833, 0.8740258926, 0.9239905856, 0.0346148639, 0.3255387878, 0.7443564112,
		0.0930828560, 0.9324878301, 0.6552622891, 0.8305292319, 0.9515416240, 0.3653033185,
		0.0504231590, 0.8494357051, 0.0908431573, 0.1567913571, 0.1211327459, 0.5269402648,
	};
	std::vector<double> data3 = {
		0.7337234864, 0.8450250437, 0.0507189845, 0.3380472189, 0.8024848119, 0.7459583505,
		0.2284865796, 0.1801980249, 0.7544559936, 0.6563679706, 0.2774781414, 0.7505901549,
		0.5510430193, 0.5049274633, 0.5092842413, 0.1561237874, 0.0534285259, 0.0532025873,
		0.8582970171, 0.7204149643, 0.2353877684, 0.5085232728, 0.6655741337, 0.5997891975,
	};
	std::vector<double> data4 = {
		0.3064594866, 0.4344703765, 0.8666062417, 0.6045467007, 0.3525326345, 0.8195923041,
		0.4241294246, 0.9263280721, 0.2202472835, 0.3833501740, 0.9746447422, 0.5632132985,
		0.3476846500, 0.5749850792, 0.2079384573, 0.2430141807, 0.9181506135, 0.9460769713,
		0.9187040562, 0.6489064808, 0.6291720783, 0.7215608801, 0.3516883305, 0.0525108831,
	};

	nnary_elementary({data, data2, data3, data4}, slist,
		[&](size_t i)
		{
			return data[i] + data2[i] + data3[i] + data4[i];
		},
		[&](size_t gradi, size_t i)
		{
			return gradi < 4 ? 1 : 0;
		});
}


TEST(API, Dense)
{
	teq::DimT ninput = 6, noutput = 5, ninput2 = 7;
	auto x = eteq::make_variable_scalar<float>(0, teq::Shape({ninput, 2}), "x");
	auto x2 = eteq::make_variable_scalar<float>(0, teq::Shape({ninput2, 2}), "x2");

	teq::TensptrT weight = eteq::make_variable_scalar<float>(0, teq::Shape({noutput, ninput}), "weight");
	teq::TensptrT bias = eteq::make_variable_scalar<float>(0, teq::Shape({noutput}), "bias");

	teq::TensptrT weight2 = eteq::make_variable_scalar<float>(0, teq::Shape({6, ninput2}), "weight");

	auto biasedy = tenncor().layer.dense(eteq::ETensor(x),
		eteq::ETensor(weight), eteq::ETensor(bias));
	auto y = tenncor().layer.dense(eteq::ETensor(x2),
		eteq::ETensor(weight2));

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(CONTRACT<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", biasedy);

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(CONTRACT<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:x2<FLOAT>[7\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(variable:weight<FLOAT>[6\\7\\1\\1\\1\\1\\1\\1])", y);
}


TEST(API, DenseSerialization)
{
	onnx::ModelProto model;

	teq::DimT ninput = 6, noutput = 5;
	std::vector<float> weight_data;
	std::vector<float> bias_data;
	{
		auto x = eteq::make_variable_scalar<float>(0, teq::Shape({ninput, 2}), "x");
		eteq::VarptrT weight = eteq::make_variable_scalar<float>(
			0, teq::Shape({noutput, ninput}), "weight");
		eteq::VarptrT bias = eteq::make_variable_scalar<float>(
			0, teq::Shape({noutput}), "bias");
		auto y = tenncor().layer.dense(eteq::ETensor(x),
			eteq::ETensor(weight), eteq::ETensor(bias));
		eteq::VarptrsT contents = layr::get_storage(y);
		ASSERT_EQ(2, contents.size());
		EXPECT_ARRHAS(contents, weight);
		EXPECT_ARRHAS(contents, bias);

		float* w = (float*) weight->device().data();
		float* b = (float*) bias->device().data();
		weight_data = std::vector<float>(w, w + weight->shape().n_elems());
		bias_data = std::vector<float>(b, b + bias->shape().n_elems());
		EXPECT_GRAPHEQ(
			"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(CONTRACT<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", y);

		tcr::save_model(model, {y});
	}
	ASSERT_EQ(noutput * ninput, weight_data.size());
	ASSERT_EQ(noutput, bias_data.size());
	{
		// load
		onnx::TensptrIdT ids;
		auto roots = tcr::load_model(ids, model);
		ASSERT_EQ(1, roots.size());

		EXPECT_GRAPHEQ(
			"(IDENTITY<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_`--(ADD<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(CONTRACT<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:x<FLOAT>[6\\2\\1\\1\\1\\1\\1\\1])\n"
			"_____|___`--(variable:weight<FLOAT>[5\\6\\1\\1\\1\\1\\1\\1])\n"
			"_____`--(EXTEND<FLOAT>[5\\2\\1\\1\\1\\1\\1\\1])\n"
			"_________`--(variable:bias<FLOAT>[5\\1\\1\\1\\1\\1\\1\\1])", roots.front());
	}
}


TEST(API, Conv)
{
	auto x = eteq::make_variable_scalar<float>(
		0, teq::Shape({4, 10, 9, 2}), "x");

	std::pair<teq::DimT,teq::DimT> filters = {6, 5};
	teq::DimT indim = 4;
	teq::DimT outdim = 3;
	teq::TensptrT weight = eteq::make_variable_scalar<float>(0,
		teq::Shape({outdim, indim, filters.second, filters.first}), "weight");
	teq::TensptrT bias = eteq::make_variable_scalar<float>(0, teq::Shape({outdim}), "bias");
	auto y = tenncor().layer.conv2d(eteq::ETensor(x),
		eteq::ETensor(weight), eteq::ETensor(bias));

	EXPECT_GRAPHEQ(
		"(IDENTITY<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_`--(ADD<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____`--(PERMUTE<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_____|___`--(CONV<FLOAT>[1\\6\\4\\2\\3\\1\\1\\1])\n"
		"_____|_______`--(PAD<FLOAT>[4\\10\\9\\2\\5\\1\\1\\1])\n"
		"_____|_______|___`--(variable:x<FLOAT>[4\\10\\9\\2\\1\\1\\1\\1])\n"
		"_____|_______`--(REVERSE<FLOAT>[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____|___________`--(variable:weight<FLOAT>[3\\4\\5\\6\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<FLOAT>[3\\6\\4\\2\\1\\1\\1\\1])\n"
		"_________`--(variable:bias<FLOAT>[3\\1\\1\\1\\1\\1\\1\\1])", y);
}


#endif // DISABLE_TENNCOR_API_TEST
