#include "gtest/gtest.h"

#include "testutil/common.hpp"

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

static const Range<double> default_range = {-9876, 9876};


struct API : public TestModel {};


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
	verify(out, shape, data);

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
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
	auto dest = op(src,src2);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
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

	auto dest2 = op(src,src);
	auto gsame = dest2->gradient(src);
	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsame.get());
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
	auto dest = op(src,src2);

	llo::GenericData out = llo::evaluate(llo::INT32, dest.get());
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

	auto dest2 = op(src,src);
	auto gsame = dest2->gradient(src);
	llo::GenericData gout = llo::evaluate(llo::INT32, gsame.get());
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
	uint8_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});

	unary_generic(sess, default_range,
	[dim](ade::Tensorptr& src) { return llo::flip(src, dim); },
	[dim, &sess](llo::GenericData& out, ade::Shape& shape, std::vector<double>& data)
	{
		auto expectshape = shape.as_list();
		auto gotshape = out.shape_.as_list();
		ASSERT_ARREQ(expectshape, gotshape);
		double* optr = (double*) out.data_.get();
		size_t n = data.size();

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
	},
	[](double* gout, std::vector<double>& og)
	{
		for (size_t i = 0, n = og.size(); i < n; ++i)
		{
			EXPECT_EQ(1, gout[i]);
		}
	});
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
	
}


TEST_F(API, Matmul)
{
	SESSION sess = get_session("API::Matmul");
	
}


TEST_F(API, Permute)
{
	SESSION sess = get_session("API::Permute");
	
}


TEST_F(API, Extend)
{
	SESSION sess = get_session("API::Extend");
	
}


TEST_F(API, Reshape)
{
	SESSION sess = get_session("API::Reshape");
	
}


#endif /* DISABLE_API_TEST */
