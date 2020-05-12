
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/isession.hpp"

#include "utils/coord.hpp"

#include "eteq/make.hpp"
#include "eteq/derive.hpp"

#include "generated/api.hpp"


using UnaryDblF = std::function<double(double)>;

template <typename T>
using UnaryOpF = std::function<eteq::ETensor<T>(eteq::ETensor<T>&)>;

template <typename T>
using BinaryOpF = std::function<eteq::ETensor<T>(eteq::ETensor<T>&,eteq::ETensor<T>&)>;

template <typename T>
using LhsBinaryOpF = std::function<eteq::ETensor<T>(eteq::ETensor<T>&,T&)>;

template <typename T>
using RhsBinaryOpF = std::function<eteq::ETensor<T>(T&,eteq::ETensor<T>&)>;

template <typename T>
using BinaryFwdF = std::function<T(T,T)>;

template <typename T>
using BinaryBwdF = std::function<T(T,T,T,T)>;

using MatVecT = std::vector<std::vector<int32_t>>;

static const int FREIVALD_N = 10;


static MatVecT create_2d (eteq::ETensor<int32_t> data,
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
		std::uniform_int_distribution<int> dist{0, 1};
		std::generate(r.begin(), r.end(), [&] { return dist(eigen::default_engine()); });

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
	std::function<void(eteq::ETensor<double>,teq::Shape&,std::vector<double>&)> verify,
	std::function<void(double*,std::vector<double>&)> bwverify)
{
	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = op(src);

	if (auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get()))
	{
		eigen::default_device().calc(*dtens);
		eigen::default_device().calc(*dtens); // idempotency check
	}
	verify(dest, shape, data);

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gsrc = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gsrc.begin(), gsrc.end()));
	session.update();
	session.update(); // idempotency check

	auto gotshape = gsrc[0]->shape();
	ASSERT_ARREQ(shape, gotshape);
	double* goptr = (double*) gsrc[0]->device().data();
	bwverify(goptr, data);
}


static void unar_elem (std::vector<double> data,
	std::vector<teq::DimT> shape_list,
	UnaryOpF<double> op, UnaryDblF fwd, UnaryDblF bwd)
{
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = op(src);
	eteq::ETensor<double> uninit_dest = op(src);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	{
		auto op_sess = eigen::get_session();
		op_sess.track({dest});
		op_sess.update();
		op_sess.update(); // idempotency check
	}
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
	}

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gsrc = eteq::derive(uninit_dest, {src});

	session.track(teq::TensptrsT(gsrc.begin(), gsrc.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gsrc[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gsrc[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
	}
}


static void unary_elementary (UnaryOpF<double> op,
	UnaryDblF fwd, UnaryDblF bwd)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	unar_elem(data, slist, op, fwd, bwd);

	// matrix optimized operation
	std::vector<teq::DimT> slist_2d = {2, 3};
	std::vector<double> data_2d = {
		59, 10, 28,
		10, 67, 62,
	};
	unar_elem(data_2d, slist_2d, op, fwd, bwd);
}


static void binar_elem (std::vector<double> data, std::vector<double> data2,
	std::vector<teq::DimT> shape_list, BinaryOpF<double> op,
	LhsBinaryOpF<double> lhs_op, RhsBinaryOpF<double> rhs_op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd, double cst)
{
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	assert(data.size() == n);
	assert(data2.size() == n);

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> src2 = eteq::make_constant<double>(data2.data(), shape);
	eteq::ETensor<double> dest = op(src, src2);
	eteq::ETensor<double> clhs = lhs_op(src, cst);
	eteq::ETensor<double> crhs = rhs_op(cst, src2);

	auto session = eigen::get_session();
	session.track({dest});
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* optr = (double*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	session.track({clhs, crhs});
	session.update();
	session.update(); // idempotency check
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

	eteq::ETensor<double> dest2 = op(src, src);
	eteq::ETensorsT<double> gsame = eteq::derive(dest2, {src});
	session.track(teq::TensptrsT(gsame.begin(), gsame.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gsame[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gsame[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1., 1.), goptr[i]);
	}

	eteq::ETensorsT<double> gleft = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gleft[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr2 = (double*) gleft[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1., 0.), goptr2[i]);
	}

	eteq::ETensorsT<double> gright = eteq::derive(dest, {src2});
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gright[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr3 = (double*) gright[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0., 1.), goptr3[i]);
	}
}


static void binary_elementary (BinaryOpF<double> op,
	LhsBinaryOpF<double> lhs_op, RhsBinaryOpF<double> rhs_op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	// tensor operation
	std::vector<teq::DimT> slist = {3, 2, 4};
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
	std::vector<teq::DimT> slist_2d = {3, 2};
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
	std::vector<teq::DimT> shape_list, BinaryOpF<int32_t> op,
	LhsBinaryOpF<int32_t> lhs_op, RhsBinaryOpF<int32_t> rhs_op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd, int32_t cst)
{
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	eteq::ETensor<int32_t> src = eteq::make_constant<int32_t>(data.data(), shape);
	eteq::ETensor<int32_t> src2 = eteq::make_constant<int32_t>(data2.data(), shape);
	eteq::ETensor<int32_t> dest = op(src, src2);
	eteq::ETensor<int32_t> clhs = lhs_op(src, cst);
	eteq::ETensor<int32_t> crhs = rhs_op(cst, src2);

	auto session = eigen::get_session();
	session.track({dest});
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* optr = (int32_t*) dest->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	session.track({clhs, crhs});
	session.update();
	session.update(); // idempotency check
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

	eteq::ETensor<int32_t> dest2 = op(src, src);
	eteq::ETensorsT<int32_t> gsame = eteq::derive(dest2, {src});
	session.track(teq::TensptrsT(gsame.begin(), gsame.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gsame[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr = (int32_t*) gsame[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data[i], 1., 1.), goptr[i]);
	}

	eteq::ETensorsT<int32_t> gleft = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gleft[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gleft[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 1., 0.), goptr2[i]);
	}

	eteq::ETensorsT<int32_t> gright = eteq::derive(dest, {src2});
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gright[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gright[0]->device().data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 0., 1.), goptr3[i]);
	}
}


static void binary_elementary_int (BinaryOpF<int32_t> op,
	LhsBinaryOpF<int32_t> lhs_op, RhsBinaryOpF<int32_t> rhs_op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	// tensor operation
	std::vector<teq::DimT> slist = {4, 3, 2};
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
	std::vector<teq::DimT> slist_2d = {4, 2};
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
	std::vector<teq::DimT> shape_list,
	std::function<double(size_t)> calc_expect,
	std::function<double(size_t,size_t)> calc_grad)
{
	teq::Shape shape(shape_list);
	teq::NElemT n = shape.n_elems();

	eteq::ETensorsT<double> srcs;
	srcs.reserve(datas.size());
	for (auto& data : datas)
	{
		assert(data.size() == n);
		srcs.push_back(eteq::make_constant<double>(data.data(), shape));
	}
	eteq::ETensor<double> dest = tenncor<double>().sum(srcs);

	auto session = eigen::get_session();

	session.track({dest});
	session.update();
	session.update(); // idempotency check
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

	for (size_t i = 0, m = datas.size(); i < m; ++i)
	{
		eteq::ETensorsT<double> gsrc = eteq::derive(dest, {srcs[i]});
		session.track(teq::TensptrsT(gsrc.begin(), gsrc.end()));
		session.update();
		session.update(); // idempotency check
		{
			auto gotshape = gsrc[0]->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr = (double*) gsrc[0]->device().data();
		for (size_t j = 0; j < n; ++j)
		{
			double expect = calc_grad(i, j);
			EXPECT_DOUBLE_EQ(expect, goptr[j]);
		}
	}
}


TEST(API, Assign)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
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

	eteq::EVariable<double> target1 = eteq::make_variable<double>(data.data(), shape);
	eteq::EVariable<double> target2 = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor<double> src = eteq::make_constant<double>(data2.data(), shape);

	auto ass1 = tenncor<double>().assign(target1, src);
	auto ass2 = tenncor<double>().assign(target2, -src);

	eigen::default_device().calc(*ass1);
	eigen::default_device().calc(*ass1); // idempotency check
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

	auto session = eigen::get_session();
	session.track({ass2});
	session.update();
	session.update(); // idempotency check
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

	EXPECT_FATAL(eteq::derive(ass1, {src}), "cannot derive ASSIGN");
	EXPECT_FATAL(eteq::derive(ass2, {src}), "cannot derive ASSIGN");
}


TEST(API, Identity)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().identity(tenncor<double>().abs(a)); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); });
}


TEST(API, Depends)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
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

	eteq::EVariable<double> target = eteq::make_variable<double>(data.data(), shape);
	eteq::ETensor<double> a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;

	auto ass = tenncor<double>().depends(tenncor<double>().assign(target, b), {c});

	auto session = eigen::get_session();
	session.track({ass});
	session.update();
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

	EXPECT_FATAL(eteq::derive(ass, {a}), "Unknown op DEPEND");
}


// ensures depend's observee is executed only once
TEST(API, DependsRunOnce)
{
	// tensor operation
	std::vector<teq::DimT> slist = {2, 3, 4};
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

	eteq::EVariable<double> target = eteq::make_variable_scalar<double>(0, shape);
	eteq::ETensor<double> a = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> b = eteq::make_constant<double>(data2.data(), shape);
	auto c = a + b;

	// assign add is non-idempotent
	auto ass = tenncor<double>().depends(tenncor<double>().assign_add(target, b), {c});

	auto session = eigen::get_session();
	session.track({ass});
	session.update();
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
		[](eteq::ETensor<double>& a) { return tenncor<double>().abs(a); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); });
}


TEST(API, Neg)
{
	auto fwd = [](double d) { return -d; };
	auto bwd = [](double d) { return -1.; };
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().neg(a); },
		fwd, bwd);
	unary_elementary(
		[](eteq::ETensor<double>& a) { return -a; },
		fwd, bwd);
}


TEST(API, Sin)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().sin(a); },
		[](double d) { return std::sin(d); },
		[](double d) { return std::cos(d); });
}


TEST(API, Cos)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().cos(a); },
		[](double d) { return std::cos(d); },
		[](double d) { return -std::sin(d); });
}


TEST(API, Tan)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().tan(a); },
		[](double d) { return std::tan(d); },
		[](double d) {
			double denom = std::cos(d);
			return 1. / denom / denom;
		});
}


TEST(API, Exp)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().exp(a); },
		[](double d) { return std::exp(d); },
		[](double d) { return std::exp(d); });
}


TEST(API, Log)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().log(a); },
		[](double d) { return std::log(d); },
		[](double d) { return 1. / d; });
}


TEST(API, Sqrt)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().sqrt(a); },
		[](double d) { return std::sqrt(d); },
		[](double d) { return 1. / (2 * std::sqrt(d)); });
}


TEST(API, Round)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().round(a); },
		[](double d) { return std::round(d); },
		[](double d) { return 1.; });
}


TEST(API, Sigmoid)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().sigmoid(a); },
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
		[](eteq::ETensor<double>& a) { return tenncor<double>().tanh(a); },
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
		[](eteq::ETensor<double>& a) { return tenncor<double>().square(a); },
		[](double d) { return d * d; },
		[](double d) { return 2 * d; });
}


TEST(API, Cube)
{
	unary_elementary(
		[](eteq::ETensor<double>& a) { return tenncor<double>().cube(a); },
		[](double d) { return d * d * d; },
		[](double d) { return 3 * d * d; });
}


TEST(API, Pow)
{
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().pow(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().pow(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().pow(a, b); },
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
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().add(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().add(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().add(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return a + b; },
		[](eteq::ETensor<double>& a, double& b)
		{ return a + b; },
		[](double& a, eteq::ETensor<double>& b)
		{ return a + b; },
		fwd, bwd);
}


TEST(API, Sub)
{
	auto fwd = [](double a, double b) { return a - b; };
	auto bwd = [](double a, double b, double leftg, double rightg)
		{ return leftg - rightg; };
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().sub(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().sub(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().sub(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return a - b; },
		[](eteq::ETensor<double>& a, double& b)
		{ return a - b; },
		[](double& a, eteq::ETensor<double>& b)
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
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().mul(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().mul(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().mul(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return a * b; },
		[](eteq::ETensor<double>& a, double& b)
		{ return a * b; },
		[](double& a, eteq::ETensor<double>& b)
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
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().div(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().div(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().div(a, b); },
		fwd, bwd);
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return a / b; },
		[](eteq::ETensor<double>& a, double& b)
		{ return a / b; },
		[](double& a, eteq::ETensor<double>& b)
		{ return a / b; },
		fwd, bwd);
}


TEST(API, Min)
{
	binary_elementary(
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().min(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().min(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().min(a, b); },
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
		[](eteq::ETensor<double>& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().max(a, b); },
		[](eteq::ETensor<double>& a, double& b)
		{ return tenncor<double>().max(a, b); },
		[](double& a, eteq::ETensor<double>& b)
		{ return tenncor<double>().max(a, b); },
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
	std::vector<teq::DimT> slist = {3, 2, 4};
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
	std::vector<teq::DimT> slist_2d = {3, 2};
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
		std::vector<teq::DimT> shape_list)
	{
		teq::Shape shape(shape_list);
		teq::NElemT n = shape.n_elems();

		assert(data.size() == n);
		assert(data2.size() == n);

		eteq::ETensor<double> cond_src =
			eteq::make_constant<double>(cond.data(), shape);
		eteq::ETensor<double> src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor<double> src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor<double> dest =
			tenncor<double>().if_then_else(cond_src, src, src2);

		auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
		ASSERT_NE(nullptr, dtens);
		eigen::default_device().calc(*dtens);
		eigen::default_device().calc(*dtens); // idempotency check
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

		auto session = eigen::get_session();

		eteq::ETensor<double> dest2 = tenncor<double>().if_then_else(cond_src, src, src);
		EXPECT_EQ(dest2.get(), src.get());

		eteq::ETensorsT<double> gleft = eteq::derive(dest, {src});
		session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
		session.update();
		session.update(); // idempotency check
		{
			auto gotshape = gleft[0]->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr = (double*) gleft[0]->device().data();
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(cond[i], goptr[i]);
		}

		eteq::ETensorsT<double> gright = eteq::derive(dest, {src2});
		session.track(teq::TensptrsT(gright.begin(), gright.end()));
		session.update();
		session.update(); // idempotency check
		{
			auto gotshape = gright[0]->shape();
			ASSERT_ARREQ(shape, gotshape);
		}
		double* goptr2 = (double*) gright[0]->device().data();
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
	std::vector<teq::DimT> slist = {3, 2, 4};
	std::vector<double> data = {
		0.0919361505, 0.5135099474, 0.3147548326, 0.0281299379, 0.3705218798, 0.6808164860,
		0.1933972592, 0.2326945471, 0.4600163558, 0.1600801317, 0.9942654588, 0.8739832345,
		0.9664644529, 0.6152766955, 0.8795922916, 0.6384690466, 0.3922073677, 0.5979097486,
		0.0425608731, 0.1178122813, 0.1594330664, 0.0926580999, 0.9309809737, 0.2119471989,
	};

	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);

	eteq::ETensor<double> src =
		eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = tenncor<double>().slice(src, {{0, 3}, {1, 1}});

	teq::Shape exshape({3, 1, 4});
	std::vector<double> exdata = {
		0.0281299379, 0.3705218798, 0.6808164860,
		0.1600801317, 0.9942654588, 0.8739832345,
		0.6384690466, 0.3922073677, 0.5979097486,
		0.0926580999, 0.9309809737, 0.2119471989,
	};
	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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
	auto session = eigen::get_session();
	eteq::ETensorsT<double> g = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(g.begin(), g.end()));
	session.update();
	session.update(); // idempotency check
	auto gotshape2 = g[0]->shape();
	ASSERT_ARREQ(shape, gotshape2);
	double* goptr = (double*) g[0]->device().data();
	std::vector<double> gotdata2(goptr, goptr + gotshape2.n_elems());
	EXPECT_VECEQ(exdata2, gotdata2);
}


TEST(API, Eq)
{
	auto fwd = [](int32_t a, int32_t b) { return a == b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().eq(a, b); },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return tenncor<int32_t>().eq(a, b); },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().eq(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return a == b; },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return a == b; },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return a == b; },
		fwd, bwd);
}


TEST(API, Neq)
{
	auto fwd = [](int32_t a, int32_t b) { return a != b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().neq(a, b); },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return tenncor<int32_t>().neq(a, b); },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().neq(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return a != b; },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return a != b; },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return a != b; },
		fwd, bwd);
}


TEST(API, Lt)
{
	auto fwd = [](int32_t a, int32_t b) { return a < b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().lt(a, b); },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return tenncor<int32_t>().lt(a, b); },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().lt(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return a < b; },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return a < b; },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return a < b; },
		fwd, bwd);
}


TEST(API, Gt)
{
	auto fwd = [](int32_t a, int32_t b) { return a > b; };
	auto bwd = [](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{ return 0; };
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().gt(a, b); },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return tenncor<int32_t>().gt(a, b); },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return tenncor<int32_t>().gt(a, b); },
		fwd, bwd);
	binary_elementary_int(
		[](eteq::ETensor<int32_t>& a, eteq::ETensor<int32_t>& b)
		{ return a > b; },
		[](eteq::ETensor<int32_t>& a, int32_t& b)
		{ return a > b; },
		[](int32_t& a, eteq::ETensor<int32_t>& b)
		{ return a > b; },
		fwd, bwd);
}


TEST(API, NElems)
{
	unary_generic(
		[](eteq::ETensor<double>& src) { return tenncor<double>().n_elems(src); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>&)
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
		[dim](eteq::ETensor<double>& src) { return tenncor<double>().n_dims(src, dim); },
		[dim](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>&)
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
	teq::Shape shape({2, 3, 4});
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = tenncor<double>().argmax(src);

	if (auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get()))
	{
		eigen::default_device().calc(*dtens);
		eigen::default_device().calc(*dtens); // idempotency check
	}
	teq::Shape oshape = dest->shape();
	teq::Shape exshape;
	EXPECT_ARREQ(exshape, oshape);
	double* ptr = (double*) dest->device().data();
	EXPECT_EQ(8, *ptr);

	EXPECT_FATAL(eteq::derive(dest, {src}), "cannot derive ARGMAX");
}


TEST(API, Rsum)
{
	unary_generic(
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_sum(src); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
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
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_sum(src, 1, 1); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
		{
			std::vector<teq::DimT> expect_list(shape.begin(), shape.end());
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
	std::vector<teq::DimT> slist = {2, 2, 3};
	teq::Shape shape(slist);
	std::vector<int32_t> data = {
		2, 1,
		7, 3,

		6, 9,
		6, 8,

		9, 7,
		7, 2,
	};

	eteq::ETensor<int32_t> src = eteq::make_constant<int32_t>(data.data(), shape);
	eteq::ETensor<int32_t> dest = tenncor<int32_t>().reduce_prod(src);
	eteq::ETensor<int32_t> dest2 = tenncor<int32_t>().reduce_prod(src, 1, 1);

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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
	eigen::default_device().calc(*dtens2);
	eigen::default_device().calc(*dtens2); // idempotency check
	{
		std::vector<teq::DimT> expect_list(shape.begin(), shape.end());
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

	auto session = eigen::get_session();

	eteq::ETensor<int32_t> gsrc = eteq::derive(dest, {src})[0];
	eteq::ETensor<int32_t> gsrc2 = eteq::derive(dest2, {src})[0];
	session.track({gsrc, gsrc2});
	session.update();
	session.update(); // idempotency check

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
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_min(src); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
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
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_min(src, 1, 1); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
		{
			std::vector<teq::DimT> expect_list(shape.begin(), shape.end());
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
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_max(src); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
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
		[](eteq::ETensor<double>& src) { return tenncor<double>().reduce_max(src, 1, 1); },
		[](eteq::ETensor<double> out, teq::Shape& shape, std::vector<double>& data)
		{
			std::vector<teq::DimT> expect_list(shape.begin(), shape.end());
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
	std::vector<teq::DimT> slist = {4, 3, 2};
	std::vector<teq::RankT> pidx = {2, 0, 1};
	teq::Shape shape(slist);
	teq::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		70, 36, 93, 50, 59, 98, 39, 5, 54, 84, 100, 94,
		75, 64, 30, 17, 90, 79, 21, 54, 6, 7, 69, 53
	};

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = tenncor<double>().permute(src, pidx);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gsrc = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gsrc.begin(), gsrc.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gsrc[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gsrc[0]->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST(API, Extend)
{
	std::vector<teq::DimT> slist = {2, 5};
	std::vector<teq::DimT> ext = {1, 3};
	teq::Shape shape(slist);
	teq::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		51, 42, 9, 43, 37, 36, 65, 95, 10, 33
	};

	eteq::ETensor<double> src = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> dest = tenncor<double>().extend(src, slist.size(), ext);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gsrc = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gsrc.begin(), gsrc.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gsrc[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr = (double*) gsrc[0]->device().data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(ext_nelem, goptr[i]);
	}
}


TEST(API, Matmul)
{
	std::vector<teq::DimT> alist = {3, 2};
	std::vector<teq::DimT> blist = {4, 3};
	std::vector<teq::DimT> sqrlist = {3, 3};
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

	eteq::ETensor<int32_t> a = eteq::make_constant<int32_t>(data.data(), ashape);
	eteq::ETensor<int32_t> b = eteq::make_constant<int32_t>(data2.data(), bshape);
	eteq::ETensor<int32_t> dest = tenncor<int32_t>().matmul(a, b);

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
	teq::Shape gotshape = dest->shape();
	EXPECT_EQ(4, gotshape.at(0));
	EXPECT_EQ(2, gotshape.at(1));
	int32_t* optr = (int32_t*) dest->device().data();
	ASSERT_NE(nullptr, optr);

	MatVecT dda = create_2d(a);
	MatVecT ddb = create_2d(b);
	MatVecT ddc = create_2d(dest);
	EXPECT_TRUE(freivald(dda, ddb, ddc));

	auto session = eigen::get_session();

	eteq::ETensor<int32_t> c = eteq::make_constant<int32_t>(data3.data(), cshape);
	eteq::ETensor<int32_t> dest2 = tenncor<int32_t>().matmul(c, c);
	eteq::ETensorsT<int32_t> gsame = eteq::derive(dest2, {c});
	session.track(teq::TensptrsT(gsame.begin(), gsame.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gcshape = gsame[0]->shape();
	ASSERT_ARREQ(cshape, gcshape);

	eteq::ETensorsT<int32_t> gleft = eteq::derive(dest, {a});
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gashape = gleft[0]->shape();
	{
		ASSERT_ARREQ(ashape, gashape);
		int32_t* ga = (int32_t*) gleft[0]->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT<int32_t> gright = eteq::derive(dest, {b});
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gbshape = gright[0]->shape();
	{
		ASSERT_ARREQ(bshape, gbshape);
		int32_t* gb = (int32_t*) gright[0]->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


TEST(API, Contract)
{
	std::vector<teq::DimT> alist = {3, 1, 2};
	std::vector<teq::DimT> blist = {4, 1, 3};
	std::vector<teq::DimT> sqrlist = {3, 2};
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

	eteq::ETensor<int32_t> a = eteq::make_constant<int32_t>(data.data(), ashape);
	eteq::ETensor<int32_t> b = eteq::make_constant<int32_t>(data2.data(), bshape);
	eteq::ETensor<int32_t> dest = tenncor<int32_t>().contract(a, b, {{0, 2}});

	auto dtens = dynamic_cast<eteq::Functor<int32_t>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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

	auto session = eigen::get_session();

	eteq::ETensor<int32_t> c = eteq::make_constant<int32_t>(data3.data(), cshape);
	eteq::ETensor<int32_t> dest2 = tenncor<int32_t>().contract(c, c, {{0, 0}});
	eteq::ETensorsT<int32_t> gsame = eteq::derive(dest2, {c});
	session.track(teq::TensptrsT(gsame.begin(), gsame.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gcshape = gsame[0]->shape();
	ASSERT_ARREQ(cshape, gcshape);

	eteq::ETensorsT<int32_t> gleft = eteq::derive(dest, {a});
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gashape = gleft[0]->shape();
	{
		ASSERT_ARREQ(ashape, gashape);
		int32_t* ga = (int32_t*) gleft[0]->device().data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT<int32_t> gright = eteq::derive(dest, {b});
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	teq::Shape gbshape = gright[0]->shape();
	{
		ASSERT_ARREQ(bshape, gbshape);
		int32_t* gb = (int32_t*) gright[0]->device().data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


static void test_rand_unif (std::vector<teq::DimT> shape_list)
{
	double hi = 3.2234;
	double lo = 0.2547977589;
	teq::Shape shape(shape_list);

	eteq::ETensor<double> src = eteq::make_constant_scalar<double>(lo, shape);
	eteq::ETensor<double> src2 = eteq::make_constant_scalar<double>(hi, shape);
	eteq::ETensor<double> dest = tenncor<double>().random.rand_unif(src, src2);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
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

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gleft = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gleft[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr2 = (double*) gleft[0]->device().data();
	EXPECT_DOUBLE_EQ(0, goptr2[0]);

	eteq::ETensorsT<double> gright = eteq::derive(dest, {src});
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gotshape = gright[0]->shape();
		ASSERT_ARREQ(shape, gotshape);
	}
	double* goptr3 = (double*) gright[0]->device().data();
	EXPECT_DOUBLE_EQ(0, goptr3[0]);
}


TEST(API, RandUniform)
{
	// tensor operation
	std::vector<teq::DimT> slist = {31, 21, 14};
	test_rand_unif(slist);

	// matrix optimized operation
	std::vector<teq::DimT> slist_2d = {31, 14};
	test_rand_unif(slist_2d);
}


TEST(API, Convolution)
{
	std::vector<teq::DimT> alist = {2, 4, 3, 3};
	std::vector<teq::DimT> blist = {1, 2, 2, 1};
	teq::Shape shape(alist);
	teq::Shape kshape(blist);
	std::vector<teq::DimT> expectslist = {
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

	eteq::ETensor<double> img = eteq::make_constant<double>(data.data(), shape);
	eteq::ETensor<double> kernel = eteq::make_constant<double>(data2.data(), kshape);
	std::vector<teq::RankT> dims(teq::rank_cap);
	std::iota(dims.begin(), dims.end(), 0);
	eteq::ETensor<double> dest = tenncor<double>().convolution(img, kernel, dims);

	auto dtens = dynamic_cast<eteq::Functor<double>*>(dest.get());
	ASSERT_NE(nullptr, dtens);
	eigen::default_device().calc(*dtens);
	eigen::default_device().calc(*dtens); // idempotency check
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(expectslist, gotshape);

		double* optr = (double*) dest->device().data();
		ASSERT_NE(nullptr, optr);
		std::vector<double> outdata(optr, optr + gotshape.n_elems());
		ASSERT_VECEQ(expect_out, outdata);
	}

	auto session = eigen::get_session();

	eteq::ETensorsT<double> gleft = eteq::derive(dest, {img});
	ASSERT_NE(nullptr, gleft[0].get());
	session.track(teq::TensptrsT(gleft.begin(), gleft.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gashape = gleft[0]->shape();
		ASSERT_ARREQ(shape, gashape);
		double* ga = (double*) gleft[0]->device().data();
		std::vector<double> ga_data(ga, ga + gashape.n_elems());
		ASSERT_VECEQ(expect_ga, ga_data);
	}

	eteq::ETensorsT<double> gright = eteq::derive(dest, {kernel});
	ASSERT_NE(nullptr, gright[0].get());
	session.track(teq::TensptrsT(gright.begin(), gright.end()));
	session.update();
	session.update(); // idempotency check
	{
		auto gbshape = gright[0]->shape();
		ASSERT_ARREQ(kshape, gbshape);
		double* gb = (double*) gright[0]->device().data();
		std::vector<double> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_VECEQ(expect_gb, gb_data);
	}
}


TEST(API, GroupSum)
{
	// tensor operation
	std::vector<teq::DimT> slist = {3, 2, 4};
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


#endif // DISABLE_API_TEST
