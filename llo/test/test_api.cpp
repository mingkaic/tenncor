#include "gtest/gtest.h"

#include "testutil/common.hpp"
#include "retroc/rand.hpp"

#include "llo/api.hpp"


#ifndef DISABLE_API_TEST


using UNARY_DBL = std::function<double(double)>;
using UNARY_OP = std::function<ade::Tensorptr(ade::Tensorptr&)>;
using BINARY_OP = std::function<ade::Tensorptr(ade::Tensorptr&,
	ade::Tensorptr&)>;

template <typename T>
using BINARY_FWD = std::function<T(T,T)>;

template <typename T>
using BINARY_BWD = std::function<T(T,T,T,T)>;

using TWODV = std::vector<std::vector<int32_t>>;

static const Range<double> default_range = {-9876, 9876};

const int FREIVALD_N = 10;


struct API : public TestModel {};


TWODV create_2d (llo::GenericData& data)
{
	int32_t* ptr = (int32_t*) data.data_.get();
	std::vector<ade::DimT> dims = data.shape_.as_list();
	ade::DimT C = dims[0];
	ade::DimT R = dims[1];
	TWODV res;

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


bool freivald (TWODV a, TWODV b, TWODV c)
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
		std::vector<int32_t> r = get_vec<int32_t>(bdim, {0, 1});

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


static void unary_generic (SESSION& sess,
	Range<double> range, UNARY_OP op,
	std::function<void(llo::GenericData&,ade::Shape&,std::vector<double>&)> verify,
	std::function<void(double*,std::vector<double>&)> bwverify)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = op(src);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	verify(out, shape, data);

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	bwverify(goptr, data);
}


static void unary_elementary (SESSION& sess,
	Range<double> range, UNARY_OP op,
	UNARY_DBL fwd, UNARY_DBL bwd, bool save_grad = true)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = op(src);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = out.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* optr = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(fwd(data[i]), optr[i]);
		}
	});

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	auto verify = [&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i]), goptr[i]);
		}
	};
	if (false == save_grad)
	{
		verify();
	}
	else
	{
		double_verify(sess, "gout", std::vector<double>(goptr, goptr + n), verify);
	}
}


static void binary_elementary (SESSION& sess,
	Range<double> range, BINARY_OP op,
	BINARY_FWD<double> fwd, BINARY_BWD<double> bwd)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, range);
	std::vector<double> data2 = sess->get_double("data2", n, range);

	auto src = llo::Source<double>::get(shape, data);
	auto src2 = llo::Source<double>::get(shape, data2);
	auto dest = op(src, src2);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = out.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* optr = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(fwd(data[i], data2[i]), optr[i]);
		}
	});

	auto dest2 = op(src, src);
	auto gsame = dest2->gradient(src);
	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsame.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	double_verify(sess, "gout", std::vector<double>(goptr, goptr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data[i], 1.0, 1.0), goptr[i]);
		}
	});

	auto gleft = dest->gradient(src);
	llo::GenericData gout_left = llo::evaluate(llo::DOUBLE, gleft.get());
	ASSERT_EQ(llo::DOUBLE, gout_left.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout_left.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr2 = (double*) gout_left.data_.get();
	double_verify(sess, "gout_left", std::vector<double>(goptr2, goptr2 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 1.0, 0.0), goptr2[i]);
		}
	});

	auto gright = dest->gradient(src2);
	llo::GenericData gout_right = llo::evaluate(llo::DOUBLE, gright.get());
	ASSERT_EQ(llo::DOUBLE, gout_right.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout_right.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr3 = (double*) gout_right.data_.get();
	double_verify(sess, "gout_right", std::vector<double>(goptr3, goptr3 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(bwd(data[i], data2[i], 0.0, 1.0), goptr3[i]);
		}
	});
}


static void binary_elementary_int (SESSION& sess,
	Range<int32_t> range, BINARY_OP op,
	BINARY_FWD<int32_t> fwd, BINARY_BWD<int32_t> bwd)
{
	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<int32_t> data = sess->get_int("data", n, range);
	std::vector<int32_t> data2 = sess->get_int("data2", n, range);

	auto src = llo::Source<int32_t>::get(shape, data);
	auto src2 = llo::Source<int32_t>::get(shape, data2);
	auto dest = op(src, src2);

	llo::GenericData out = llo::evaluate(llo::INT32, dest.get());
	ASSERT_EQ(llo::INT32, out.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = out.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* optr = (int32_t*) out.data_.get();
	int_verify(sess, "out", std::vector<int32_t>(optr, optr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
		}
	});

	auto dest2 = op(src, src);
	auto gsame = dest2->gradient(src);
	llo::GenericData gout = llo::evaluate(llo::INT32, gsame.get());
	ASSERT_EQ(llo::INT32, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr = (int32_t*) gout.data_.get();
	int_verify(sess, "gout", std::vector<int32_t>(goptr, goptr + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data[i], 1, 1), goptr[i]);
		}
	});

	auto gleft = dest->gradient(src);
	llo::GenericData gout_left = llo::evaluate(llo::INT32, gleft.get());
	ASSERT_EQ(llo::INT32, gout_left.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout_left.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gout_left.data_.get();
	int_verify(sess, "gout_left", std::vector<int32_t>(goptr2, goptr2 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data2[i], 1, 0), goptr2[i]);
		}
	});

	auto gright = dest->gradient(src2);
	llo::GenericData gout_right = llo::evaluate(llo::INT32, gright.get());
	ASSERT_EQ(llo::INT32, gout_right.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout_right.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gout_right.data_.get();
	int_verify(sess, "gout_right", std::vector<int32_t>(goptr3, goptr3 + n),
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(bwd(data[i], data2[i], 0, 1), goptr3[i]);
		}
	});
}


TEST_F(API, Abs)
{
	SESSION sess = get_session("API::Abs");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::abs(a); },
	[](double d) { return std::abs(d); },
	[](double d) { return 1.0; }, false);
}


TEST_F(API, Neg)
{
	SESSION sess = get_session("API::Neg");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::neg(a); },
	[](double d) { return -d; },
	[](double d) { return -1.0; }, false);
}


TEST_F(API, Not)
{
	SESSION sess = get_session("API::Not");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::bit_not(a); },
	[](double d) { return !d; },
	[](double d) { return !1.0; }, false);
}


TEST_F(API, Sin)
{
	SESSION sess = get_session("API::Sin");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::sin(a); },
	[](double d) { return std::sin(d); },
	[](double d) { return std::cos(d); });
}


TEST_F(API, Cos)
{
	SESSION sess = get_session("API::Cos");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::cos(a); },
	[](double d) { return std::cos(d); },
	[](double d) { return -std::sin(d); });
}


TEST_F(API, Tan)
{
	SESSION sess = get_session("API::Tan");
	unary_elementary(sess, {-1, 1},
	[](ade::Tensorptr& a) { return llo::tan(a); },
	[](double d) { return std::tan(d); },
	[](double d) {
		double denom = std::cos(d);
		return 1.0 / denom / denom;
	});
}


TEST_F(API, Exp)
{
	SESSION sess = get_session("API::Exp");
	unary_elementary(sess, {-9876, 5},
	[](ade::Tensorptr& a) { return llo::exp(a); },
	[](double d) { return std::exp(d); },
	[](double d) { return std::exp(d); });
}


TEST_F(API, Log)
{
	SESSION sess = get_session("API::Log");
	unary_elementary(sess, {0.5, 9876},
	[](ade::Tensorptr& a) { return llo::log(a); },
	[](double d) { return std::log(d); },
	[](double d) { return 1.0 / d; });
}


TEST_F(API, Sqrt)
{
	SESSION sess = get_session("API::Sqrt");
	unary_elementary(sess, {0, 9876},
	[](ade::Tensorptr& a) { return llo::sqrt(a); },
	[](double d) { return std::sqrt(d); },
	[](double d) { return 1.0 / (2 * std::sqrt(d)); });
}


TEST_F(API, Round)
{
	SESSION sess = get_session("API::Round");
	unary_elementary(sess, default_range,
	[](ade::Tensorptr& a) { return llo::round(a); },
	[](double d) { return std::round(d); },
	[](double d) { return 1.0; }, false);
}


TEST_F(API, Flip)
{
	SESSION sess = get_session("API::Flip");

	int32_t nrank = sess->get_scalar("nrank", {1, ade::rank_cap - 1});
	std::vector<ade::DimT> slist = get_shape_n(sess, nrank, "shape");
	ade::Shape shape(slist);
	uint8_t dim = 0;
	if (nrank > 1)
	{
		dim = sess->get_scalar("dim", {0, nrank - 1});
	}
	uint8_t baddim = sess->get_scalar("baddim", {nrank, ade::rank_cap});
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::flip(src, dim);

	auto bad = llo::flip(src, baddim);
	EXPECT_THROW(llo::evaluate(llo::DOUBLE, bad.get()), std::runtime_error) <<
		"baddim: " << (int) baddim << " nrank: " << nrank;

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	auto expectshape = shape.as_list();
	auto gotshape = out.shape_.as_list();
	ASSERT_ARREQ(expectshape, gotshape);
	double* optr = (double*) out.data_.get();

	double_verify(sess, "out", std::vector<double>(optr, optr + n),
	[&]()
	{
		std::vector<ade::DimT> coord;
		uint8_t dimlimit = shape.at(dim) - 1;
		for (size_t i = 0; i < n; ++i)
		{
			coord = ade::coordinate(shape, i);
			coord[dim] = dimlimit - coord[dim];

			EXPECT_EQ(data[ade::index(shape, coord)], optr[i]);
		}
	});

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST_F(API, Pow)
{
	SESSION sess = get_session("API::Pow");
	binary_elementary(sess, {0, 5},
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::pow(a, b); },
	[](double a, double b) { return std::pow(a, b); },
	[](double a, double b, double leftg, double rightg)
	{
		return std::pow(a, b - 1) * (leftg * b + rightg * a * std::log(a));
	});
}


TEST_F(API, Add)
{
	SESSION sess = get_session("API::Add");
	binary_elementary(sess, default_range,
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::add(a, b); },
	[](double a, double b) { return a + b; },
	[](double a, double b, double leftg, double rightg)
	{
		return leftg + rightg;
	});
}


TEST_F(API, Sub)
{
	SESSION sess = get_session("API::Sub");
	binary_elementary(sess, default_range,
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::sub(a, b); },
	[](double a, double b) { return a - b; },
	[](double a, double b, double leftg, double rightg)
	{
		return leftg - rightg;
	});
}


TEST_F(API, Mul)
{
	SESSION sess = get_session("API::Mul");
	binary_elementary(sess, default_range,
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::mul(a, b); },
	[](double a, double b) { return a * b; },
	[](double a, double b, double leftg, double rightg)
	{
		return leftg * a + rightg * b;
	});
}


TEST_F(API, Div)
{
	SESSION sess = get_session("API::Div");
	binary_elementary(sess, default_range,
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::div(a, b); },
	[](double a, double b) { return a / b; },
	[](double a, double b, double leftg, double rightg)
	{
		return (leftg * b - rightg * a) / (b * b);
	});
}


TEST_F(API, Eq)
{
	SESSION sess = get_session("API::Eq");
	binary_elementary_int(sess, {-1, 1},
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::eq(a, b); },
	[](int32_t a, int32_t b) { return a == b; },
	[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
	{
		return leftg == rightg;
	});
}


TEST_F(API, Neq)
{
	SESSION sess = get_session("API::Neq");
	binary_elementary_int(sess, {-1, 1},
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::neq(a, b); },
	[](int32_t a, int32_t b) { return a != b; },
	[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
	{
		return leftg != rightg;
	});
}


TEST_F(API, Lt)
{
	SESSION sess = get_session("API::Lt");
	binary_elementary_int(sess, {-1, 1},
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::lt(a, b); },
	[](int32_t a, int32_t b) { return a < b; },
	[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
	{
		return leftg < rightg;
	});
}


TEST_F(API, Gt)
{
	SESSION sess = get_session("API::Gt");
	binary_elementary_int(sess, {-1, 1},
	[](ade::Tensorptr& a, ade::Tensorptr& b) { return llo::gt(a, b); },
	[](int32_t a, int32_t b) { return a > b; },
	[](int32_t a, int32_t b, int32_t leftg, int32_t rightg)
	{
		return leftg > rightg;
	});
}


TEST_F(API, NElems)
{
	SESSION sess = get_session("API::NElems");

	unary_generic(sess, default_range,
	[](ade::Tensorptr& src) { return llo::n_elems(src); },
	[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>&)
	{
		EXPECT_EQ(0, out.shape_.n_rank());
		ASSERT_EQ(1, out.shape_.n_elems());
		double got = *((double*) out.data_.get());

		double_verify(sess, "out", {got},
		[&]()
		{
			EXPECT_EQ(shape.n_elems(), got);
		});
	},
	[](double* gout, std::vector<double>& og)
	{
		for (size_t i = 0, n = og.size(); i < n; ++i)
		{
			EXPECT_EQ(0, gout[i]);
		}
	});
}


TEST_F(API, NDims)
{
	SESSION sess = get_session("API::NDims");
	uint8_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});

	unary_generic(sess, default_range,
	[dim](ade::Tensorptr& src) { return llo::n_dims(src, dim); },
	[dim, &sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>&)
	{
		EXPECT_EQ(0, out.shape_.n_rank());
		ASSERT_EQ(1, out.shape_.n_elems());
		double got = *((double*) out.data_.get());

		double_verify(sess, "out", {got},
		[&]()
		{
			EXPECT_EQ(shape.at(dim), got);
		});
	},
	[](double* gout, std::vector<double>& og)
	{
		for (size_t i = 0, n = og.size(); i < n; ++i)
		{
			EXPECT_EQ(0, gout[i]);
		}
	});
}


TEST_F(API, Argmax)
{
	SESSION sess = get_session("API::Argmax");

	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);
	auto it = data.begin();
	auto maxit = std::max_element(it, data.end());
	size_t maxidx = std::distance(it, maxit);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::argmax(src);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	EXPECT_EQ(llo::DOUBLE, out.dtype_);
	EXPECT_EQ(0, out.shape_.n_rank());
	ASSERT_EQ(1, out.shape_.n_elems());
	double got = *((double*) out.data_.get());
	double_verify(sess, "out", {got},
	[&]()
	{
		EXPECT_EQ(maxidx, got);
	});

	EXPECT_THROW(dest->gradient(src), std::bad_function_call);
}


TEST_F(API, Rmax)
{
	SESSION sess = get_session("API::Rmax");

	unary_generic(sess, default_range,
	[](ade::Tensorptr& src) { return llo::rmax(src); },
	[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>& data)
	{
		size_t n = out.shape_.n_elems();
		EXPECT_EQ(0, out.shape_.n_rank());
		ASSERT_EQ(1, n);
		double got = *((double*) out.data_.get());

		double_verify(sess, "out", {got},
		[&]()
		{
			double expect = *(std::max_element(data.begin(), data.end()));
			EXPECT_DOUBLE_EQ(expect, got);
		});
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
		}
	});
}


TEST_F(API, Rsum)
{
	SESSION sess = get_session("API::Rsum");

	unary_generic(sess, default_range,
	[](ade::Tensorptr& src) { return llo::rsum(src); },
	[&sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>& data)
	{
		size_t n = out.shape_.n_elems();
		{
			EXPECT_EQ(0, out.shape_.n_rank());
			ASSERT_EQ(1, n);
		}
		double got = *((double*) out.data_.get());

		double_verify(sess, "out", {got},
		[&]()
		{
			double expect = std::accumulate(data.begin(), data.end(), 0.0);
			EXPECT_DOUBLE_EQ(expect, got);
		});
	},
	[](double* gout, std::vector<double>& og)
	{
		for (size_t i = 0, n = og.size(); i < n; ++i)
		{
			EXPECT_EQ(1, gout[i]);
		}
	});
}


TEST_F(API, Matmul2d)
{
	SESSION sess = get_session("API::Matmul2d");

	ade::DimT cdim = sess->get_scalar("cdim", {1, 17});
	ade::DimT adim = sess->get_scalar("adim", {1, 17});
	ade::DimT bdim = sess->get_scalar("bdim", {1, 13});
	std::vector<ade::DimT> alist = {cdim, adim};
	std::vector<ade::DimT> blist = {bdim, cdim};
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Shape cshape({cdim, cdim});

	ade::NElemT na = ashape.n_elems();
	ade::NElemT nb = bshape.n_elems();
	std::vector<int32_t> data = sess->get_int("data", na, {-9876, 9876});
	std::vector<int32_t> data2 = sess->get_int("data2", nb, {-9876, 9876});
	std::vector<int32_t> data3 = sess->get_int("data3", cdim * cdim, {-9876, 9876});

	auto a = llo::Source<int32_t>::get(ashape, data);
	auto b = llo::Source<int32_t>::get(bshape, data2);
	auto dest = llo::matmul(a, b);

	llo::GenericData out = llo::evaluate(llo::INT32, dest.get());
	EXPECT_EQ(llo::INT32, out.dtype_);
	ade::Shape& gotshape = out.shape_;
	EXPECT_EQ(2, gotshape.n_rank());
	EXPECT_EQ(bdim, gotshape.at(0));
	EXPECT_EQ(adim, gotshape.at(1));
	int32_t* optr = (int32_t*) out.data_.get();
	int_verify(sess, "out",
	std::vector<int32_t>(optr, optr + gotshape.n_elems()),
	[&]()
	{
		llo::GenericData ad = static_cast<llo::iSource*>(a.get())->
			evaluate(llo::INT32);
		llo::GenericData bd = static_cast<llo::iSource*>(b.get())->
			evaluate(llo::INT32);
		TWODV dda = create_2d(ad);
		TWODV ddb = create_2d(bd);
		TWODV ddc = create_2d(out);
		EXPECT_TRUE(freivald(dda, ddb, ddc));
	});

	auto c = llo::Source<int32_t>::get(cshape, data3);
	auto dest2 = llo::matmul(c, c);
	auto gsame = dest2->gradient(c);
	llo::GenericData gout = llo::evaluate(llo::INT32, gsame.get());
	EXPECT_EQ(llo::INT32, gout.dtype_);
	ade::Shape& gcshape = gout.shape_;
	{
		std::vector<ade::DimT> expectshape = {cdim, cdim, cdim, cdim};
		auto gotshape = gcshape.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr = (int32_t*) gout.data_.get();

	// int_verify(sess, "gout",
	// std::vector<int32_t>(goptr, goptr + gcshape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });

	auto gleft = dest->gradient(a);
	llo::GenericData gout_left = llo::evaluate(llo::INT32, gleft.get());
	EXPECT_EQ(llo::INT32, gout_left.dtype_);
	ade::Shape& gashape = gout_left.shape_;
	{
		auto expectshape = gotshape.as_list();
		expectshape.insert(expectshape.end(),
			alist.begin(), alist.end());

		auto gotshape = gashape.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr2 = (int32_t*) gout_left.data_.get();

	// int_verify(sess, "gout_left",
	// std::vector<int32_t>(goptr2, goptr2 + gashape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });

	auto gright = dest->gradient(b);
	llo::GenericData gout_right = llo::evaluate(llo::INT32, gright.get());
	EXPECT_EQ(llo::INT32, gout_right.dtype_);
	ade::Shape& gbshape = gout_right.shape_;
	{
		auto expectshape = gotshape.as_list();
		expectshape.insert(expectshape.end(),
			blist.begin(), blist.end());

		auto gotshape = gbshape.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	int32_t* goptr3 = (int32_t*) gout_right.data_.get();

	// int_verify(sess, "gout_right",
	// std::vector<int32_t>(goptr3, goptr3 + gbshape.n_elems()),
	// [&]()
	// {
	// 	// todo: implement
	// });
}


TEST_F(API, DISABLED_Matmul)
{
	SESSION sess = get_session("API::Matmul");

}


TEST_F(API, Permute)
{
	SESSION sess = get_session("API::Permute");

	int32_t nrank = sess->get_scalar("nrank", {2, ade::rank_cap - 2});
	std::vector<ade::DimT> slist = get_shape_n(sess, nrank, "slist");
	std::vector<uint64_t> pidx_temp = sess->choose("pidx", slist.size(), slist.size());
	std::vector<uint8_t> pidx(pidx_temp.begin(), pidx_temp.end());
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = sess->get_double("data", nelem, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::permute(src, pidx);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	size_t n = out.shape_.n_elems();
	ASSERT_EQ(nelem, n);
	double* got = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>{got, got + n},
	[&]()
	{
		std::vector<ade::DimT> coord;
		std::vector<ade::DimT> temp;
		for (size_t i = 0; i < n; ++i)
		{
			coord = temp = ade::coordinate(shape, i);
			for (int32_t j = 0; j < nrank; ++j)
			{
				coord[j] = temp[pidx[j]];
			}

			EXPECT_EQ(data[i], got[ade::index(out.shape_, coord)]);
		}
	});

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0, n = data.size(); i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST_F(API, Extend)
{
	SESSION sess = get_session("API::Extend");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");

	int32_t nrank = slist.size();
	int32_t remainder = ade::rank_cap - nrank;

	int32_t n_ext = 1;
	if (remainder > 1)
	{
		n_ext = sess->get_scalar("n_ext", {1, remainder});
	}
	std::vector<ade::DimT> ext = get_shape_n(sess, n_ext, "ext");
	ade::Shape shape(slist);
	ade::NElemT nelem = shape.n_elems();
	std::vector<double> data = sess->get_double("data", nelem, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::extend(src, ext);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	size_t ext_nelem = ade::Shape(ext).n_elems();
	size_t n = out.shape_.n_elems();
	ASSERT_EQ(nelem * ext_nelem, n);
	double* got = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>{got, got + n},
	[&]()
	{
		for (size_t i = 0; i < nelem; ++i)
		{
			for (size_t j = 0; j < ext_nelem; ++j)
			{
				EXPECT_EQ(data[i], got[i + j * nelem]);
			}
		}
	});

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


TEST_F(API, Reshape)
{
	SESSION sess = get_session("API::Reshape");

	int32_t nrank = sess->get_scalar("nrank", {2, ade::rank_cap - 2});
	std::vector<ade::DimT> slist = get_shape_n(sess, nrank, "slist");
	uint8_t mergeidx = 0;
	if (nrank > 2)
	{
		mergeidx = sess->get_scalar("mergeidx", {0, (uint8_t) nrank - 2});
	}
	std::vector<ade::DimT> olist = slist;
	olist.erase(olist.begin() + mergeidx);
	olist[mergeidx] *= slist[mergeidx];

	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::reshape(src, olist);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	ASSERT_EQ(llo::DOUBLE, out.dtype_);
	ASSERT_EQ(n, out.shape_.n_elems());
	double* got = (double*) out.data_.get();
	double_verify(sess, "out", std::vector<double>{got, got + n},
	[&]()
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(data[i], got[i]);
		}
	});

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	ASSERT_EQ(llo::DOUBLE, gout.dtype_);
	ASSERT_EQ(n, gout.shape_.n_elems());
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


#endif /* DISABLE_API_TEST */
