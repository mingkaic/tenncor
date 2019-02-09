
#ifndef DISABLE_API_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

// #include "bwd/grader.hpp"

#include "ead/generated/api.hpp"
#include "ead/variable.hpp"
// #include "ead/opt/derive.hpp"


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


MatVecT create_2d (ead::TensMapT<int32_t>* data)
{
	int32_t* ptr = (int32_t*) data->data();
	ade::DimT C = data->dimension(0);
	ade::DimT R = data->dimension(1);
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


bool freivald (MatVecT a, MatVecT b, MatVecT c)
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
	std::function<void(ead::TensMapT<double>*,ade::Shape&,std::vector<double>&)> verify,
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

	auto out = dest->get_tensmap();
	dest->update();
	verify(out, shape, data);

	// ead::Session session;

	// ead::NodeptrT<double> gsrc = ead::derive(dest, src.get());
	// auto gout = gsrc->get_tensmap();
	// session.track(gsrc);
	// session.update();

	// auto gotshape = gout->dimensions();
	// ASSERT_ARREQ(slist, gotshape);
	// double* goptr = (double*) gout->data();
	// bwverify(goptr, data);
}


static void unary_elementary (UnaryOpF<double> op,
	UnaryDblF fwd, UnaryDblF bwd)
{
	std::vector<ade::DimT> slist = {2, 3, 4};
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> dest = op(src);

	auto out = dest->get_tensmap();
	dest->update();
	{
		auto gotshape = out->dimensions();
		ASSERT_ARREQ(slist, gotshape);
	}
	double* optr = (double*) out->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
	}

	// ead::Session session;

	// ead::NodeptrT<double> gsrc = ead::derive(dest, src.get());
	// auto gout = gsrc->get_tensmap();
	// session.track(gsrc);
	// session.update();
	// {
	// 	auto gotshape = gout->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr = (double*) gout->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
	// }
}


static void binary_elementary (BinaryOpF<double> op,
	BinaryFwdF<double> fwd, BinaryBwdF<double> bwd)
{
	std::vector<ade::DimT> slist = {3, 2, 4};
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
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

	ead::NodeptrT<double> src = ead::make_constant<double>(data.data(), shape);
	ead::NodeptrT<double> src2 = ead::make_constant<double>(data2.data(), shape);
	ead::NodeptrT<double> dest = op(src, src2);

	auto out = dest->get_tensmap();
	dest->update();
	{
		auto gotshape = out->dimensions();
		ASSERT_ARREQ(slist, gotshape);
	}
	double* optr = (double*) out->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	// ead::Session session;

	// ead::NodeptrT<double> dest2 = op(src, src);
	// ead::NodeptrT<double> gsame = ead::derive(dest2, src.get());
	// auto gout = gsame->get_tensmap();
	// session.track(gsame);
	// session.update();
	// {
	// 	auto gotshape = gout->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr = (double*) gout->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
	// }

	// ead::NodeptrT<double> gleft = ead::derive(dest, src.get());
	// auto gout_left = gleft->get_tensmap();
	// session.track(gout_left);
	// session.update();
	// {
	// 	auto gotshape = gout_left->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr2 = (double*) gout_left->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
	// }

	// ead::NodeptrT<double> gright = ead::derive(dest, src2.get());
	// ead::NodeptrT<double> gout_right = gright->get_tensmap();
	// session.track(gout_right);
	// session.update();
	// {
	// 	auto gotshape = gout_right->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr3 = (double*) gout_right->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
	// }
}


static void binary_elementary_int (BinaryOpF<int32_t> op,
	BinaryFwdF<int32_t> fwd, BinaryBwdF<int32_t> bwd)
{
	std::vector<ade::DimT> slist = {4, 3, 2};
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<int32_t> data = {
		1, 2, 3, 0, 1, 2, 2, 1, 1, 3, 3, 1,
		2, 2, 3, 0, 1, 3, 3, 1, 2, 0, 0, 2
	};
	std::vector<int32_t> data2 = {
		0, 0, 2, 1, 3, 3, 2, 2, 3, 1, 2, 3,
		1, 3, 1, 3, 1, 0, 2, 1, 2, 2, 0, 1
	};

	ead::NodeptrT<int32_t> src = ead::make_constant<int32_t>(data.data(), shape);
	ead::NodeptrT<int32_t> src2 = ead::make_constant<int32_t>(data2.data(), shape);
	ead::NodeptrT<int32_t> dest = op(src, src2);

	auto out = dest->get_tensmap();
	dest->update();
	{
		auto gotshape = out->dimensions();
		ASSERT_ARREQ(slist, gotshape);
	}
	int32_t* optr = (int32_t*) out->data();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
	}

	// ead::Session session;

	// ead::NodeptrT<int32_t> dest2 = op(src, src);
	// ead::NodeptrT<int32_t> gsame = ead::derive(dest2, src.get());
	// auto gout = gsame->get_tensmap();
	// session.track(gsame);
	// session.update();
	// {
	// 	auto gotshape = gout->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// int32_t* goptr = (int32_t*) gout->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
	// }

	// ead::NodeptrT<int32_t> gleft = ead::derive(dest, src.get());
	// auto gout_left = gleft->get_tensmap();
	// session.track(gout_left);
	// session.update();
	// {
	// 	auto gotshape = gout_left->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// int32_t* goptr2 = (int32_t*) gout_left->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
	// }

	// ead::NodeptrT<int32_t> gright = ead::derive(dest, src2.get());
	// ead::NodeptrT<int32_t> gout_right = gright->get_tensmap();
	// session.track(gout_right);
	// session.update();
	// {
	// 	auto gotshape = gout_right->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// int32_t* goptr3 = (int32_t*) gout_right->data();
	// for (size_t i = 0; i < n; ++i)
	// {
	// 	EXPECT_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
	// }
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
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, ead::get_shape<double>(*out).n_elems());
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
		[dim](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>&)
		{
			ASSERT_EQ(1, ead::get_shape<double>(*out).n_elems());
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
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = ead::get_shape<double>(*out).n_elems();
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
		[](ead::NodeptrT<double>& src) { return age::reduce_sum(src, 1, 2); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = ead::get_shape<double>(*out);
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
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::reduce_prod(src); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = ead::get_shape<double>(*out).n_elems();
			{
				ASSERT_EQ(1, n);
			}
			double got = *((double*) out->data());

			double expect = std::accumulate(data.begin(), data.end(), 1.0, std::multiplies<double>());
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
		[](ead::NodeptrT<double>& src) { return age::reduce_prod(src, 1, 2); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = ead::get_shape<double>(*out);
			EXPECT_ARREQ(expect_list, gotshape);

			ade::CoordT coord;
			ade::DimT d = shape.at(1);
			double* got = (double*) out->data();
			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				coord = ade::coordinate(gotshape, i);
				double acc = 1;
				for (size_t j = 0; j < d; ++j)
				{
					coord[1] = j;
					acc *= data[ade::index(shape, coord)];
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


TEST(API, Rmin)
{
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::reduce_min(src); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = ead::get_shape<double>(*out).n_elems();
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
		[](ead::NodeptrT<double>& src) { return age::reduce_min(src, 1, 2); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = ead::get_shape<double>(*out);
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
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(1, gout[i]);
			}
		});
}


TEST(API, Rmax)
{
	unary_generic(
		[](ead::NodeptrT<double>& src) { return age::reduce_max(src); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			size_t n = ead::get_shape<double>(*out).n_elems();
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
		[](ead::NodeptrT<double>& src) { return age::reduce_max(src, 1, 2); },
		[](ead::TensMapT<double>* out, ade::Shape& shape, std::vector<double>& data)
		{
			std::vector<ade::DimT> expect_list(shape.begin(), shape.end());
			expect_list[1] = 1;
			ade::Shape gotshape = ead::get_shape<double>(*out);
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
			for (size_t i = 0, n = og.size(); i < n; ++i)
			{
				EXPECT_EQ(1, gout[i]);
			}
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

	auto out = dest->get_tensmap();
	dest->update();
	size_t n = ead::get_shape<double>(*out).n_elems();
	ASSERT_EQ(nelem, n);
	double* got = (double*) out->data();
	ade::CoordT coord, temp;
	for (size_t i = 0; i < n; ++i)
	{
		coord = temp = ade::coordinate(shape, i);
		for (int32_t j = 0, n = slist.size(); j < n; ++j)
		{
			coord[j] = temp[pidx[j]];
		}

		EXPECT_EQ(data[i], got[ade::index(ead::get_shape<double>(*out), coord)]);
	}

	// ead::Session session;

	// ead::NodeptrT<double> gsrc = ead::derive(dest, src.get());
	// auto gout = gsrc->get_tensmap();
	// session.track(gsrc);
	// session.update();
	// {
	// 	auto gotshape = gout->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr = (double*) gout->data();
	// for (size_t i = 0, n = data.size(); i < n; ++i)
	// {
	// 	EXPECT_EQ(1, goptr[i]);
	// }
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

	auto out = dest->get_tensmap();
	dest->update();
	size_t ext_nelem = ade::Shape(ext).n_elems();
	size_t n = ead::get_shape<double>(*out).n_elems();
	ASSERT_EQ(nelem * ext_nelem, n);
	double* got = (double*) out->data();
	for (size_t i = 0; i < nelem; ++i)
	{
		for (size_t j = 0; j < ext_nelem; ++j)
		{
			EXPECT_EQ(data[i], got[i + j * nelem]);
		}
	}

	// ead::Session session;

	// ead::NodeptrT<double> gsrc = ead::derive(dest, src.get());
	// auto gout = gsrc->get_tensmap();
	// session.track(gsrc);
	// session.update();
	// {
	// 	auto gotshape = gout->dimensions();
	// 	ASSERT_ARREQ(slist, gotshape);
	// }
	// double* goptr = (double*) gout->data();
	// for (size_t i = 0, n = data.size(); i < n; ++i)
	// {
	// 	EXPECT_EQ(ext_nelem, goptr[i]);
	// }
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

	auto out = dest->get_tensmap();
	dest->update();
	ade::Shape gotshape = ead::get_shape<int32_t>(*out);
	EXPECT_EQ(4, gotshape.at(0));
	EXPECT_EQ(2, gotshape.at(1));
	int32_t* optr = (int32_t*) out->data();
	ASSERT_NE(nullptr, optr);

	auto ad = a->get_tensmap();
	auto bd = b->get_tensmap();
	MatVecT dda = create_2d(ad);
	MatVecT ddb = create_2d(bd);
	MatVecT ddc = create_2d(out);
	EXPECT_TRUE(freivald(dda, ddb, ddc));

	// ead::Session session;

	// ead::NodeptrT<int32_t> c = ead::make_constant<int32_t>(data3.data(), cshape);
	// ead::NodeptrT<int32_t> dest2 = age::matmul(c, c);
	// ead::NodeptrT<int32_t> gsame = llo::derive(dest2, c.get());
	// auto gout = gsame->get_tensmap();
	// session.track(gsame);
	// session.update();
	// ade::Shape gcshape = ead::get_shape<int32_t>(*gout);
	// {
	// 	std::vector<ade::DimT> glist(gcshape.begin(), gcshape.end());
	// 	ASSERT_ARREQ(sqrlist, glist);
	// }

	// ead::NodeptrT<int32_t> gleft = llo::derive(dest, a.get());
	// session.track(gleft);
	// session.update();
	// auto gout_left = gleft->get_tensmap();
	// ade::Shape gashape = ead::get_shape<int32_t>(*gout_left);
	// {
	// 	std::vector<ade::DimT> glist(gashape.begin(), gashape.end());
	// 	ASSERT_ARREQ(alist, glist);
	// 	int32_t* ga = (int32_t*) gout_left->data();
	// 	ASSERT_NE(nullptr, ga);
	// 	std::vector<int32_t> ga_data(ga, ga + gashape.n_elems());
	// 	ASSERT_ARREQ(expect_ga, ga_data);
	// }

	// ead::NodeptrT<int32_t> gright = llo::derive(dest, b.get());
	// session.track(gright);
	// session.update();
	// auto gout_right = gright->get_tensmap();
	// ade::Shape gbshape = ead::get_shape<int32_t>(*gout_right);
	// {
	// 	std::vector<ade::DimT> glist(gbshape.begin(), gbshape.end());
	// 	ASSERT_ARREQ(blist, glist);
	// 	int32_t* gb = (int32_t*) gout_right->data();
	// 	ASSERT_NE(nullptr, gb);
	// 	std::vector<int32_t> gb_data(gb, gb + gbshape.n_elems());
	// 	ASSERT_ARREQ(expect_gb, gb_data);
	// }
}


TEST(API, RandUniform)
{
	std::vector<ade::DimT> slist = {31, 21, 14};
	double hi = 3.2234;
	double lo = 0.2547977589;
	ade::Shape shape(slist);

	ead::NodeptrT<double> src = ead::make_constant<double>(lo, shape);
	ead::NodeptrT<double> src2 = ead::make_constant<double>(hi, shape);
	ead::NodeptrT<double> dest = age::rand_unif(src, src2);

	auto out = dest->get_tensmap();
	dest->update();
	{
		auto gotshape = out->dimensions();
		ASSERT_ARREQ(slist, gotshape);
	}
	double* optr = (double*) out->data();
	size_t nelems = ead::get_shape<double>(*out).n_elems();
	for (size_t i = 0; i < nelems; ++i)
	{
		EXPECT_LT(lo, optr[i]);
		EXPECT_GT(hi, optr[i]);
	}

	// ead::Session session;

	// ead::NodeptrT<double> gleft = ead::derive(dest, src.get());
	// auto gout_left = gleft->get_tensmap();
	// ASSERT_EQ(1, ead::get_shape<double>(*gout_left).n_elems());
	// double* goptr2 = (double*) gout_left->data();
	// EXPECT_DOUBLE_EQ(0, goptr2[0]);

	// ead::NodeptrT<double> gright = ead::derive(dest, src.get());
	// auto gout_right = gright->get_tensmap();
	// ASSERT_EQ(1, ead::get_shape<double>(*gout_right).n_elems());
	// double* goptr3 = (double*) gout_right->data();
	// EXPECT_DOUBLE_EQ(0, goptr3[0]);
}


// TEST(API, DISABLED_Convolution)
// {
// 	std::vector<ade::DimT> alist = {2, 4, 3, 1};
// 	std::vector<ade::DimT> blist = {1, 2, 3, 3};
// 	ade::Shape shape(alist);
// 	ade::Shape kshape(blist);
// 	std::vector<ade::DimT> expectslist = {
// 		1, 2, 1, 1, 1, 1, 1, 1,
// 	};

// 	std::vector<double> data = {
// 		1,2,3,
// 		4,5,6,
// 		7,8,9,
// 		10,11,12,

// 		13,14,15,
// 		16,17,18,
// 		19,20,21,
// 		22,23,24,
// 	};
// 	std::vector<double> data2 = {
// 		2,4,3,
// 		2,4,3,
// 		2,4,3,

// 		3,3,3,
// 		4,4,4,
// 		2,2,2,
// 	};
// 	std::vector<double> expect_out = {
// 		615,
// 		723,
// 	};
// 	std::vector<double> expect_ga = {
// 		2,4,5,
// 		6,7,5,
// 		4,3,2,
// 		4,5,7,

// 		6,6,3,
// 		3,4,4,
// 		8,6,6,
// 		4,2,2,
// 	};
// 	std::vector<double> expect_gb = {
// 		4,6,8,
// 		10,12,14,
// 		20,22,24,

// 		26,28,30,
// 		36,38,40,
// 		42,44,46,
// 	};

// 	ade::TensptrT img(llo::Variable<double>::get(data, shape));
// 	ade::TensptrT kernel(llo::Variable<double>::get(data2, kshape));
// 	ade::TensptrT dest = age::convolution(img, kernel);

// 	ead::NodeptrT<double> out = llo::eval<double>(dest);
// 	ade::Shape gotshape = ead::get_shape<double>(*out);
// 	{
// 		std::vector<ade::DimT> slist(gotshape.begin(), gotshape.end());
// 		EXPECT_ARREQ(expectslist, slist);
// 		double* optr = (double*) out->data();
// 		ASSERT_NE(nullptr, optr);
// 		std::vector<double> outdata(optr, optr + gotshape.n_elems());
// 		ASSERT_ARREQ(expect_out, outdata);
// 	}

// 	ade::TensptrT gleft = llo::derive(dest, img.get());
// 	ead::NodeptrT<double> gout_left = llo::eval<double>(gleft);
// 	ade::Shape gashape = ead::get_shape<double>(*gout_left);
// 	{
// 		std::vector<ade::DimT> glist(gashape.begin(), gashape.end());
// 		ASSERT_ARREQ(alist, glist);
// 		double* ga = (double*) gout_left->data();
// 		std::vector<double> ga_data(ga, ga + gashape.n_elems());
// 		ASSERT_ARREQ(expect_ga, ga_data);
// 	}

// 	ade::TensptrT gright = llo::derive(dest, kernel.get());
// 	ead::NodeptrT<double> gout_right = llo::eval<double>(gright);
// 	ade::Shape gbshape = ead::get_shape<double>(*gout_right);
// 	{
// 		std::vector<ade::DimT> glist(gbshape.begin(), gbshape.end());
// 		ASSERT_ARREQ(blist, glist);
// 		double* gb = (double*) gout_right->data();
// 		std::vector<double> gb_data(gb, gb + gbshape.n_elems());
// 		ASSERT_ARREQ(expect_gb, gb_data);
// 	}
// }


#endif // DISABLE_API_TEST
