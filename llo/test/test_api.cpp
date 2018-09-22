#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "llo/api.hpp"


#ifndef DISABLE_API_TEST


using UNARY_DBL = std::function<double(double)>;
using UNARY_OP = std::function<ade::Tensorptr(ade::Tensorptr&)>;
using BINARY_DBL = std::function<double(double,double)>;
using BINARY_OP = std::function<ade::Tensorptr(
	ade::Tensorptr&,ade::Tensorptr&)>;


static const Range<double> default_range = {-9876, 9876};


struct API : public TestModel {};


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
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* optr = (double*) out.data_.get();

	std::string outkey = "out";
	// if (sess->generated_input())
	// {
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i]), optr[i]);
	}
	// 	sess->store_double(outkey, std::vector<double>(optr, optr + n));
	// }
	// else
	// {
	// 	auto expect = expect_double(outkey);
	// 	std::vector<double> got(optr, optr + n);
	// 	EXPECT_ARREQ(expect, got);
	// }

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();

	std::string goutkey = "gout";
	// if (sess->generated_input() || false == save_grad)
	// {
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i]), goptr[i]);
	}
	// 	if (save_grad)
	// 	{
	// 		sess->store_double(goutkey, std::vector<double>(goptr, goptr + n));
	// 	}
	// }
	// else
	// {
	// 	// we reach here if we're expecting gout to have saved value
	// 	// which occurs if input is not generated (meaning input is saved)
	// 	// and we're saving gout originally
	// 	auto expect = expect_double(goutkey);
	// 	std::vector<double> got(goptr, goptr + n);
	// 	EXPECT_ARREQ(expect, got);
	// }
}


static void binary_elementary (SESSION& sess,
	Range<double> range, BINARY_OP op,
	BINARY_DBL fwd, BINARY_DBL bwd, bool save_grad = true)
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
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* optr = (double*) out.data_.get();

	std::string outkey = "out";
	// if (sess->generated_input())
	// {
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(fwd(data[i], data2[i]), optr[i]);
	}
	// 	sess->store_double(outkey, std::vector<double>(optr, optr + n));
	// }
	// else
	// {
	// 	auto expect = expect_double(outkey);
	// 	std::vector<double> got(optr, optr + n);
	// 	EXPECT_ARREQ(expect, got);
	// }

	auto gleft = dest->gradient(src);
	auto gleft = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();

	std::string goutkey = "gout";
	// if (sess->generated_input() || false == save_grad)
	// {
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(bwd(data[i]), goptr[i]);
	}
	// 	if (save_grad)
	// 	{
	// 		sess->store_double(goutkey, std::vector<double>(goptr, goptr + n));
	// 	}
	// }
	// else
	// {
	// 	// we reach here if we're expecting gout to have saved value
	// 	// which occurs if input is not generated (meaning input is saved)
	// 	// and we're saving gout originally
	// 	auto expect = expect_double(goutkey);
	// 	std::vector<double> got(goptr, goptr + n);
	// 	EXPECT_ARREQ(expect, got);
	// }
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
	SESSION sess = get_session("API::Round");
	uint8_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});

	std::vector<ade::DimT> slist = get_shape(sess, "shape");
	ade::Shape shape(slist);
	ade::NElemT n = shape.n_elems();
	std::vector<double> data = sess->get_double("data", n, default_range);

	auto src = llo::Source<double>::get(shape, data);
	auto dest = llo::flip(src, dim);

	llo::GenericData out = llo::evaluate(llo::DOUBLE, dest.get());
	{
		auto expectshape = shape.as_list();
		auto gotshape = out.shape_.as_list();
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* optr = (double*) out.data_.get();

	std::string outkey = "out";
	// if (sess->generated_input())
	// {
	std::vector<ade::DimT> coord;
	uint8_t dimlimit = shape.at(dim) - 1;
	for (size_t i = 0; i < n; ++i)
	{
		coord = ade::coordinate(shape, i);
		coord[dim] = dimlimit - coord[dim];

		EXPECT_EQ(data[ade::index(shape, coord)], optr[i]);
	}
	// 	sess->store_double(outkey, std::vector<double>(optr, optr + n));
	// }
	// else
	// {
	// 	auto expect = expect_double(outkey);
	// 	std::vector<double> got(optr, optr + n);
	// 	EXPECT_ARREQ(expect, got);
	// }

	auto gsrc = dest->gradient(src);

	llo::GenericData gout = llo::evaluate(llo::DOUBLE, gsrc.get());
	{
		auto expectshape = shape.as_list();
		auto gotshape = gout.shape_.as_list();
		EXPECT_ARREQ(expectshape, gotshape);
	}
	double* goptr = (double*) gout.data_.get();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, goptr[i]);
	}
}


#endif /* DISABLE_API_TEST */
