
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ead/generated/api.hpp"
#include "ead/session.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/grader.hpp"


using UnaryDblF = std::function<double(double)>;

template <typename T>
using UnaryOpF = std::function<ead::NodeptrT<T>(ead::NodeptrT<T>&)>;

template <typename T>
using BinaryOpF = std::function<ead::NodeptrT<T>(ead::NodeptrT<T>&,ead::NodeptrT<T>&)>;

template <typename T>
using BinaryFwdF = std::function<T(T,T)>;

template <typename T>
using BinaryBwdF = std::function<T(T,T,T,T)>;

using MatVecT = std::vector<std::vector<int32_t>>;

static const int FREIVALD_N = 10;


static MatVecT create_2d (ead::NodeptrT<int32_t> data)
{
	int32_t* ptr = (int32_t*) data->data();
	ade::DimT C = data->shape().at(0);
	ade::DimT R = data->shape().at(1);
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
	uint8_t cdim = b.size();
	uint8_t bdim = b[0].size();
	uint8_t adim = a.size();
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
		std::generate(r.begin(), r.end(), [&]() { return dist(ead::get_engine()); });

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
	std::function<void(ead::NodeptrT<double>,ade::Shape&,std::vector<double>&)> verify,
	std::function<void(double*,std::vector<double>&)> bwverify)
{
	std::vector<ade::DimT> slist = {2, 3, 4};
	ade::Shape shape(slist);
	std::vector<double> data = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> dest = op(src);

	dest->update();
	verify(dest, shape, data);

	ead::Session<double> session;

	ead::NodeptrT<double> gsrc = ead::derive(dest, src);
	session.track(gsrc->get_tensor().get());
	session.update();

	auto gotshape = gsrc->shape();
	ASSERT_ARREQ(slist, gotshape);
	double* goptr = (double*) gsrc->data();
	bwverify(goptr, data);
}


static void unar_elem (std::vector<double> data,
	std::vector<ade::DimT> shape_list,
	UnaryOpF<double> op, UnaryDblF fwd, UnaryDblF bwd)
{
	ade::Shape shape(shape_list);
	ade::NElemT n = shape.n_elems();
	assert(data.size() == n);

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> dest = op(src);

	dest->update();
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* optr = (double*) dest->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
	}

	ead::Session<double> session;

	ead::NodeptrT<double> gsrc = ead::derive(dest, src);

	session.track(gsrc->get_tensor().get());
	session.update();
	{
		auto gotshape = gsrc->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr = (double*) gsrc->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
	}
}


static void unary_elementary (UnaryOpF<double> op,
	UnaryDblF fwd, UnaryDblF bwd)
{
	// tensor operation
	std::vector<ade::DimT> slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	unar_elem(data, slist, op, fwd, bwd);

	// matrix optimized operation
	std::vector<ade::DimT> slist_2d = {2, 3};
	std::vector<double> data_2d = {
		59, 10, 28,
		10, 67, 62,
	};
	unar_elem(data_2d, slist_2d, op, fwd, bwd);
}


static void binar_elem (std::vector<double> data, std::vector<double> data2,
	std::vector<ade::DimT> shape_list, BinaryOpF<double> op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	ade::Shape shape(shape_list);
	ade::NElemT n = shape.n_elems();

	assert(data.size() == n);
	assert(data2.size() == n);

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> src2 = ead::make_constant<double>(data2.data(), shape);
	ead::NodeptrT<double> dest = op(src, src2);

	dest->update();
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* optr = (double*) dest->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	ead::Session<double> session;

	ead::NodeptrT<double> dest2 = op(src, src);
	ead::NodeptrT<double> gsame = ead::derive(dest2, src);
	session.track(gsame->get_tensor().get());
	session.update();
	{
		auto gotshape = gsame->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr = (double*) gsame->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
	}

	ead::NodeptrT<double> gleft = ead::derive(dest, src);
	session.track(gleft->get_tensor().get());
	session.update();
	{
		auto gotshape = gleft->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr2 = (double*) gleft->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
	}

	ead::NodeptrT<double> gright = ead::derive(dest, src2);
	session.track(gright->get_tensor().get());
	session.update();
	{
		auto gotshape = gright->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr3 = (double*) gright->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
	}
}


static void binary_elementary (BinaryOpF<double> op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	// tensor operation
	std::vector<ade::DimT> slist = {3, 2, 4};
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

	binar_elem(data, data2, slist, op, fwd, bwd);

	// matrix optimized operation
	std::vector<ade::DimT> slist_2d = {3, 2};
	std::vector<double> data_2d = {
		0.0919361505, 0.5135099474, 0.3147548326,
		0.0281299379, 0.3705218798, 0.6808164860,
	};
	std::vector<double> data2_2d = {
		0.2547977589, 0.8808089905, 0.4323663340,
		0.5710527217, 0.6207772267, 0.8574923091,
	};
	binar_elem(data_2d, data2_2d, slist_2d, op, fwd, bwd);
}


static void binar_elem_int (std::vector<int32_t> data, std::vector<int32_t> data2,
	std::vector<ade::DimT> shape_list, BinaryOpF<int32_t> op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	ade::Shape shape(shape_list);
	ade::NElemT n = shape.n_elems();

	ead::NodeptrT<int32_t> src = ead::make_constant<int32_t>(data.data(), shape);
	ead::NodeptrT<int32_t> src2 = ead::make_constant<int32_t>(data2.data(), shape);
	ead::NodeptrT<int32_t> dest = op(src, src2);

	dest->update();
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	int32_t* optr = (int32_t*) dest->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	ead::Session<int32_t> session;

	ead::NodeptrT<int32_t> dest2 = op(src, src);
	ead::NodeptrT<int32_t> gsame = ead::derive(dest2, src);
	session.track(gsame->get_tensor().get());
	session.update();
	{
		auto gotshape = gsame->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	int32_t* goptr = (int32_t*) gsame->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
	}

	ead::NodeptrT<int32_t> gleft = ead::derive(dest, src);
	session.track(gleft->get_tensor().get());
	session.update();
	{
		auto gotshape = gleft->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gleft->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
	}

	ead::NodeptrT<int32_t> gright = ead::derive(dest, src2);
	session.track(gright->get_tensor().get());
	session.update();
	{
		auto gotshape = gright->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gright->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
	}
}


static void binary_elementary_int (BinaryOpF<int32_t> op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	// tensor operation
	std::vector<ade::DimT> slist = {4, 3, 2};
	std::vector<int32_t> data = {
		1, 2, 3, 0, 1, 2, 2, 1, 1, 3, 3, 1,
		2, 2, 3, 0, 1, 3, 3, 1, 2, 0, 0, 2
	};
	std::vector<int32_t> data2 = {
		0, 0, 2, 1, 3, 3, 2, 2, 3, 1, 2, 3,
		1, 3, 1, 3, 1, 0, 2, 1, 2, 2, 0, 1
	};

	binar_elem_int(data, data2, slist, op, fwd, bwd);

	// matrix optimized operation
	std::vector<ade::DimT> slist_2d = {4, 2};
	std::vector<int32_t> data_2d = {
		1, 2, 3, 0,
		1, 2, 2, 1,
	};
	std::vector<int32_t> data2_2d = {
		0, 0, 2, 1,
		3, 3, 2, 2,
	};

	binar_elem_int(data_2d, data2_2d, slist_2d, op, fwd, bwd);
}


TEST(API, Abs)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::abs(a); },
		[](double d) { return std::abs(d); },
		[](double d) { return d / std::abs(d); });
}


TEST(API, Neg)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::neg(a); },
		[](double d) { return -d; },
		[](double d) { return -1.0; });
}


TEST(API, Sin)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::sin(a); },
		[](double d) { return std::sin(d); },
		[](double d) { return std::cos(d); });
}


TEST(API, Cos)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::cos(a); },
		[](double d) { return std::cos(d); },
		[](double d) { return -std::sin(d); });
}


TEST(API, Tan)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::tan(a); },
		[](double d) { return std::tan(d); },
		[](double d) {
			double denom = std::cos(d);
			return 1.0 / denom / denom;
		});
}


TEST(API, Exp)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::exp(a); },
		[](double d) { return std::exp(d); },
		[](double d) { return std::exp(d); });
}


TEST(API, Log)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::log(a); },
		[](double d) { return std::log(d); },
		[](double d) { return 1.0 / d; });
}


TEST(API, Sqrt)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::sqrt(a); },
		[](double d) { return std::sqrt(d); },
		[](double d) { return 1.0 / (2 * std::sqrt(d)); });
}


TEST(API, Round)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::round(a); },
		[](double d) { return std::round(d); },
		[](double d) { return 1.0; });
}


TEST(API, Sigmoid)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::sigmoid(a); },
		[](double d) { return 1 / (1 + std::exp(-d)); },
		[](double d)
		{
			double sig = 1 / (1 + std::exp(-d));
			return sig * (1 - sig);
		});
}


TEST(API, SigmoidGrad)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::sigmoid_grad(a); },
		[](double d)
		{
			double sig = 1 / (1 + std::exp(-d));
			return sig * (1 - sig);
		},
		[](double d)
		{
			double sig = 1 / (1 + std::exp(-d));
			double sig_grad = sig * (1 - sig);
			return sig_grad * (1 - 2 * sig);
		});
}


TEST(API, Tanh)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::tanh(a); },
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
		[](ead::NodeptrT<double>& a) { return age::square(a); },
		[](double d) { return d * d; },
		[](double d) { return 2 * d; });
}


TEST(API, Cube)
{
	unary_elementary(
		[](ead::NodeptrT<double>& a) { return age::cube(a); },
		[](double d) { return d * d * d; },
		[](double d) { return 3 * d * d; });
}


TEST(API, Pow)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::pow(a, b); },
		[](double a, double b) { return std::pow(a, b); },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg * b * std::pow(a, b - 1) +
				rightg * std::pow(a, b) * std::log(a);
		});
}


TEST(API, Add)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::add(a, b); },
		[](double a, double b) { return a + b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg + rightg;
		});
}


TEST(API, Sub)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::sub(a, b); },
		[](double a, double b) { return a - b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg - rightg;
		});
}


TEST(API, Mul)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::mul(a, b); },
		[](double a, double b) { return a * b; },
		[](double a, double b, double leftg, double rightg)
		{
			return leftg * b + rightg * a;
		});
}


TEST(API, Div)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::div(a, b); },
		[](double a, double b) { return a / b; },
		[](double a, double b, double leftg, double rightg)
		{
			return (leftg * b - rightg * a) / (b * b);
		});
}


TEST(API, Min)
{
	binary_elementary(
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::min(a, b); },
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
		[](ead::NodeptrT<double>& a, ead::NodeptrT<double>& b)
		{ return age::max(a, b); },
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


TEST(API, Eq)
{
	binary_elementary_int(
		[](ead::NodeptrT<int32_t>& a, ead::NodeptrT<int32_t>& b)
		{ return age::eq(a, b); },
		[](int32_t a, int32_t b) { return a == b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST(API, Neq)
{
	binary_elementary_int(
		[](ead::NodeptrT<int32_t>& a, ead::NodeptrT<int32_t>& b)
		{ return age::neq(a, b); },
		[](int32_t a, int32_t b) { return a != b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST(API, Lt)
{
	binary_elementary_int(
		[](ead::NodeptrT<int32_t>& a, ead::NodeptrT<int32_t>& b)
		{ return age::lt(a, b); },
		[](int32_t a, int32_t b) { return a < b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST(API, Gt)
{
	binary_elementary_int(
		[](ead::NodeptrT<int32_t>& a, ead::NodeptrT<int32_t>& b)
		{ return age::gt(a, b); },
		[](int32_t a, int32_t b) { return a > b; },
		[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
		{
			return 0;
		});
}


TEST(API, NElems)
{
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::n_elems(src); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out->shape().n_elems());
			double got = *((double*) out->data());

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
	uint8_t dim = 2;
	unary_generic(
		[dim](ead::NodeptrT<double>& src) { return age::n_dims(src, dim); },
		[dim](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, out->shape().n_elems());
			double got = *((double*) out->data());

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


TEST(API, Rsum)
{
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::reduce_sum(src); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			{
				ASSERT_EQ(1, n);
			}
			double got = *((double*) out->data());

			double expect = std::accumulate(data.begin(), data.end(), 0.0);
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
		[](ead::NodeptrT<double>& src) { return age::reduce_sum(src, 1, 1); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			ade::CoordT coord;
			ade::DimT d = shape.at(1);
			double* got = (double*) out->data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = ade::coordinate(gotshape, i);
				double acc = 0;
				for (size_t j = 0; j < d; ++j)
				{
					coord[1] = j;
					acc += data[ade::index(shape, coord)];
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
	std::vector<ade::DimT> slist = {2, 2, 3};
	ade::Shape shape(slist);
	std::vector<size_t> data = {
		2, 1,
		7, 3,

		6, 9,
		6, 8,

		9, 7,
		7, 2,
	};

	ead::NodeptrT<size_t> src = ead::make_constant<size_t>(data.data(), shape);
	ead::NodeptrT<size_t> dest = age::reduce_prod(src);
	ead::NodeptrT<size_t> dest2 = age::reduce_prod(src, 1, 1);

	dest->update();
	{
		size_t n = dest->shape().n_elems();
		{
			ASSERT_EQ(1, n);
		}
		size_t got = *((size_t*) dest->data());

		size_t expect = std::accumulate(data.begin(), data.end(), 1, std::multiplies<size_t>());
		EXPECT_EQ(expect, got);
	}

	dest2->update();
	{
		std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
		expect_list[1] = 1;
		ade::Shape gotshape = dest2->shape();
		EXPECT_ARREQ(expect_list, gotshape);

		ade::CoordT coord;
		ade::DimT d = shape.at(1);
		size_t* got = (size_t*) dest2->data();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			coord = ade::coordinate(gotshape, i);
			size_t acc = 1;
			for (size_t j = 0; j < d; ++j)
			{
				coord[1] = j;
				acc *= data[ade::index(shape, coord)];
			}
			EXPECT_EQ(acc, got[i]);
		}
	}

	ead::Session<size_t> session;

	ead::NodeptrT<size_t> gsrc = ead::derive(dest, src);
	ead::NodeptrT<size_t> gsrc2 = ead::derive(dest2, src);
	session.track(gsrc->get_tensor().get());
	session.track(gsrc2->get_tensor().get());
	session.update();

	auto gotshape = gsrc->shape();
	ASSERT_ARREQ(slist, gotshape);
	size_t* goptr = (size_t*) gsrc->data();
	{
		size_t n = data.size();
		std::vector<size_t> left(n, 1);
		std::vector<size_t> right(n, 1);
		for (size_t i = 1; i < n; ++i)
		{
			left[i] = data[i - 1] * left[i - 1];
			right[n - i - 1] = data[n - i] * right[n - i];
		}
		for (size_t i = 0; i < n; ++i)
		{
			size_t expect = left[i] * right[i];
			EXPECT_EQ(expect, goptr[i]);
		}
	}

	std::vector<size_t> ex_grad = {
		7, 3,
		2, 1,

		6, 8,
		6, 9,

		7, 2,
		9, 7,
	};
	auto gotshape2 = gsrc2->shape();
	ASSERT_ARREQ(slist, gotshape2);
	size_t* goptr2 = (size_t*) gsrc2->data();
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
		[](ead::NodeptrT<double>& src) { return age::reduce_min(src); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			ASSERT_EQ(1, n);
			double got = *((double*) out->data());

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
		[](ead::NodeptrT<double>& src) { return age::reduce_min(src, 1, 1); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			ade::CoordT coord;
			ade::DimT d = shape.at(1);
			double* got = (double*) out->data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = ade::coordinate(gotshape, i);
				double acc = data[ade::index(shape, coord)];
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					acc = std::min(acc, data[ade::index(shape, coord)]);
				}
				EXPECT_DOUBLE_EQ(acc, got[i]);
			}
		},
		[](double* gout, std::vector<double>& og)
		{
			ade::Shape inshape({2, 3, 4});
			ade::Shape outshape({2, 1, 4});
			ade::CoordT coord;
			ade::DimT d = 3;
			size_t m = og.size();
			size_t n = outshape.n_elems();
			std::vector<double> expect(m, 0);
			for (size_t i = 0; i < n; ++i)
			{
				coord = ade::coordinate(outshape, i);
				size_t min_idx = ade::index(inshape, coord);
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					size_t idx = ade::index(inshape, coord);
					if (og[min_idx] > og[idx])
					{
						min_idx = idx;
					}
				}
				expect[min_idx] = 1;
			}
			std::vector<double> got(gout, gout + m);
			EXPECT_ARREQ(expect, got);
		});
}


TEST(API, Rmax)
{
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::reduce_max(src); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = out->shape().n_elems();
			ASSERT_EQ(1, n);
			double got = *((double*) out->data());

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
		[](ead::NodeptrT<double>& src) { return age::reduce_max(src, 1, 1); },
		[](ead::NodeptrT<double> out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = out->shape();
			EXPECT_ARREQ(expect_list, gotshape);

			ade::CoordT coord;
			ade::DimT d = shape.at(1);
			double* got = (double*) out->data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = ade::coordinate(gotshape, i);
				double acc = data[ade::index(shape, coord)];
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					acc = std::max(acc, data[ade::index(shape, coord)]);
				}
				EXPECT_DOUBLE_EQ(acc, got[i]);
			}
		},
		[](double* gout, std::vector<double>& og)
		{
			ade::Shape inshape({2, 3, 4});
			ade::Shape outshape({2, 1, 4});
			ade::CoordT coord;
			ade::DimT d = 3;
			size_t m = og.size();
			size_t n = outshape.n_elems();
			std::vector<double> expect(m, 0);
			for (size_t i = 0; i < n; ++i)
			{
				coord = ade::coordinate(outshape, i);
				size_t max_idx = ade::index(inshape, coord);
				for (size_t j = 1; j < d; ++j)
				{
					coord[1] = j;
					size_t idx = ade::index(inshape, coord);
					if (og[max_idx] < og[idx])
					{
						max_idx = idx;
					}
				}
				expect[max_idx] = 1;
			}
			std::vector<double> got(gout, gout + m);
			EXPECT_ARREQ(expect, got);
		});
}


TEST(API, Permute)
{
	std::vector<ade::DimT> slist = {4, 3, 2};
	std::vector<uint8_t> pidx = {2, 0, 1};
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		70, 36, 93, 50, 59, 98, 39, 5, 54, 84, 100, 94,
		75, 64, 30, 17, 90, 79, 21, 54, 6, 7, 69, 53
	};

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> dest = age::permute(src, pidx);

	dest->update();
	size_t n = dest->shape().n_elems();
	ASSERT_EQ(nelem, n);
	double* got = (double*) dest->data();
	ade::CoordT coord, temp;
	for (size_t i = 0; i < n; ++i)
	{
		coord = temp = ade::coordinate(shape, i);
		for (int32_t j = 0, n = slist.size(); j < n; ++j)
		{
			coord[j] = temp[pidx[j]];
		}

		EXPECT_EQ(data[i], got[ade::index(dest->shape(), coord)]);
	}

	ead::Session<double> session;

	ead::NodeptrT<double> gsrc = ead::derive(dest, src);
	session.track(gsrc->get_tensor().get());
	session.update();
	{
		auto gotshape = gsrc->shape();
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gsrc->data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST(API, Extend)
{
	std::vector<ade::DimT> slist = {2, 5};
	std::vector<ade::DimT> ext = {1, 3};
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = {
		51, 42, 9, 43, 37, 36, 65, 95, 10, 33
	};

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> dest = age::extend(src, slist.size(), ext);

	dest->update();
	size_t ext_nelem = ade::Shape(ext).n_elems();
	size_t n = dest->shape().n_elems();
	ASSERT_EQ(nelem * ext_nelem, n);
	double* got = (double*) dest->data();
	for (size_t i = 0; i < nelem; ++i)
	{
		for (size_t j = 0; j < ext_nelem; ++j)
		{
			EXPECT_EQ(data[i], got[i + j * nelem]);
		}
	}

	ead::Session<double> session;

	ead::NodeptrT<double> gsrc = ead::derive(dest, src);
	session.track(gsrc->get_tensor().get());
	session.update();
	{
		auto gotshape = gsrc->shape();
		ASSERT_ARREQ(slist, gotshape);
	}
	double* goptr = (double*) gsrc->data();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(ext_nelem, goptr[i]);
	}
}


TEST(API, Matmul)
{
	std::vector<ade::DimT> alist = {3, 2};
	std::vector<ade::DimT> blist = {4, 3};
	std::vector<ade::DimT> sqrlist = {3, 3};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape(sqrlist);

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

	ead::NodeptrT<int32_t> a = ead::make_constant<int32_t>(data.data(), ashape);
	ead::NodeptrT<int32_t> b = ead::make_constant<int32_t>(data2.data(), bshape);
	ead::NodeptrT<int32_t> dest = age::matmul(a, b);

	dest->update();
	ade::Shape gotshape = dest->shape();
	EXPECT_EQ(4, gotshape.at(0));
	EXPECT_EQ(2, gotshape.at(1));
	int32_t* optr = (int32_t*) dest->data();
	ASSERT_NE(nullptr, optr);

	MatVecT dda = create_2d(a);
	MatVecT ddb = create_2d(b);
	MatVecT ddc = create_2d(dest);
	EXPECT_TRUE(freivald(dda, ddb, ddc));

	ead::Session<int32_t> session;

	ead::NodeptrT<int32_t> c = ead::make_constant<int32_t>(data3.data(), cshape);
	ead::NodeptrT<int32_t> dest2 = age::matmul(c, c);
	ead::NodeptrT<int32_t> gsame = ead::derive(dest2, c);
	session.track(gsame->get_tensor().get());
	session.update();
	ade::Shape gcshape = gsame->shape();
	{
		std::vector<ade::DimT> glist(gcshape.begin(), gcshape.end());
		ASSERT_ARREQ(sqrlist, glist);
	}

	ead::NodeptrT<int32_t> gleft = ead::derive(dest, a);
	session.track(gleft->get_tensor().get());
	session.update();
	ade::Shape gashape = gleft->shape();
	{
		std::vector<ade::DimT> glist(gashape.begin(), gashape.end());
		ASSERT_ARREQ(alist, glist);
		int32_t* ga = (int32_t*) gleft->data();
		ASSERT_NE(nullptr, ga);
		std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
		ASSERT_ARREQ(expect_ga, ga_data);
	}

	ead::NodeptrT<int32_t> gright = ead::derive(dest, b);
	session.track(gright->get_tensor().get());
	session.update();
	ade::Shape gbshape = gright->shape();
	{
		std::vector<ade::DimT> glist(gbshape.begin(), gbshape.end());
		ASSERT_ARREQ(blist, glist);
		int32_t* gb = (int32_t*) gright->data();
		ASSERT_NE(nullptr, gb);
		std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_ARREQ(expect_gb, gb_data);
	}
}


static void test_rand_unif (std::vector<ade::DimT> shape_list)
{
	double hi = 3.2234;
	double lo = 0.2547977589;
	ade::Shape shape(shape_list);

	ead::NodeptrT<double> src = ead::make_constant_scalar<double>(lo, shape);
	ead::NodeptrT<double> src2 = ead::make_constant_scalar<double>(hi, shape);
	ead::NodeptrT<double> dest = age::rand_unif(src, src2);

	dest->update();
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* optr = (double*) dest->data();
	size_t nelems = dest->shape().n_elems();
	for (size_t i = 0; i < nelems; ++i)
	{
		EXPECT_LT(lo, optr[i]);
		EXPECT_GT(hi, optr[i]);
	}

	ead::Session<double> session;

	ead::NodeptrT<double> gleft = ead::derive(dest, src);
	session.track(gleft->get_tensor().get());
	session.update();
	{
		auto gotshape = gleft->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr2 = (double*) gleft->data();
	EXPECT_DOUBLE_EQ(0, goptr2[0]);

	ead::NodeptrT<double> gright = ead::derive(dest, src);
	session.track(gright->get_tensor().get());
	session.update();
	{
		auto gotshape = gright->shape();
		ASSERT_ARREQ(shape_list, gotshape);
	}
	double* goptr3 = (double*) gright->data();
	EXPECT_DOUBLE_EQ(0, goptr3[0]);
}


TEST(API, RandUniform)
{
	// tensor operation
	std::vector<ade::DimT> slist = {31, 21, 14};
	test_rand_unif(slist);

	// matrix optimized operation
	std::vector<ade::DimT> slist_2d = {31, 14};
	test_rand_unif(slist_2d);
}


TEST(API, Convolution)
{
	std::vector<ade::DimT> alist = {2, 4, 3, 3};
	std::vector<ade::DimT> blist = {1, 2, 2, 1};
	ade::Shape shape(alist);
	ade::Shape kshape(blist);
	std::vector<ade::DimT> expectslist = {
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

	ead::NodeptrT<double> img = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> kernel = ead::make_constant<double>(data2.data(), kshape);
	std::vector<ade::DimT> dims(ade::rank_cap);
	std::iota(dims.begin(), dims.end(), 0);
	ead::NodeptrT<double> dest = age::convolution(img, kernel, dims);

	dest->update();
	{
		auto gotshape = dest->shape();
		ASSERT_ARREQ(expectslist, gotshape);

		double* optr = (double*) dest->data();
		ASSERT_NE(nullptr, optr);
		std::vector<double> outdata(optr, optr + gotshape.n_elems());
		ASSERT_ARREQ(expect_out, outdata);
	}

	ead::Session<double> session;

	ead::NodeptrT<double> gleft = ead::derive(dest, img);
	ASSERT_NE(nullptr, gleft);
	session.track(gleft->get_tensor().get());
	session.update();
	{
		auto gashape = gleft->shape();
		ASSERT_ARREQ(alist, gashape);
		double* ga = (double*) gleft->data();
		std::vector<double> ga_data(ga, ga + gashape.n_elems());
		ASSERT_ARREQ(expect_ga, ga_data);
	}

	ead::NodeptrT<double> gright = ead::derive(dest, kernel);
	ASSERT_NE(nullptr, gright);
	session.track(gright->get_tensor().get());
	session.update();
	{
		auto gbshape = gright->shape();
		ASSERT_ARREQ(blist, gbshape);
		double* gb = (double*) gright->data();
		std::vector<double> gb_data(gb, gb + gbshape.n_elems());
		ASSERT_ARREQ(expect_gb, gb_data);
	}
}


#endif // DISABLE_API_TEST
