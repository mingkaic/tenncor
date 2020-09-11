
#ifndef DISABLE_EIGEN_SHAPER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/eigen.hpp"


TEST(SHAPER, Default)
{
	egen::ShapeParser<egen::ABS> parser;
	marsh::Maps empty;

	EXPECT_FATAL(parser(empty, {}), eigen::no_argument_err.c_str());

	teq::ShapesT conflicting = {
		teq::Shape({3, 4, 5}),
		teq::Shape({3, 2, 5}),
	};

	EXPECT_FATAL(parser(empty, conflicting),
		"cannot ABS with incompatible shapes [3\\2\\5\\1\\1\\1\\1\\1] and [3\\4\\5\\1\\1\\1\\1\\1]");

	teq::Shape expect({3, 4, 6});
	teq::ShapesT good = {expect, expect};
	teq::Shape got = parser(empty, good);

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, Identity)
{
	//
}


TEST(SHAPER, Reduce)
{
	std::set<teq::RankT> rdims = {3, 2, 5};

	egen::ShapeParser<egen::REDUCE_SUM> parser;
	marsh::Maps dimmed;
	eigen::Packer<std::set<teq::RankT>>().pack(dimmed, rdims);

	EXPECT_FATAL(parser(dimmed, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({3, 4, 6, 7, 3});
	teq::Shape expect({3, 4, 1, 1, 3});
	teq::Shape got = parser(dimmed, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, ArgReduce)
{
	teq::RankT rdim = 3;
	egen::ShapeParser<egen::ARGMAX> parser;
	marsh::Maps dimmed;
	eigen::Packer<teq::RankT>().pack(dimmed, rdim);

	EXPECT_FATAL(parser(dimmed, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({3, 4, 6, 7, 3});

	teq::Shape expect({3, 4, 6, 1, 3});
	teq::Shape got = parser(dimmed, {inshape});
	EXPECT_ARREQ(expect, got);

	teq::RankT rdim2 = 8;
	marsh::Maps dimmed2;
	eigen::Packer<teq::RankT>().pack(dimmed2, rdim2);

	teq::Shape expect2;
	teq::Shape got2 = parser(dimmed2, {inshape});
	EXPECT_ARREQ(expect2, got2);
}


TEST(SHAPER, Permute)
{
	//
}


TEST(SHAPER, Extend)
{
	//
}


TEST(SHAPER, Reshape)
{
	teq::Shape expect({3, 4, 6});

	egen::ShapeParser<egen::RESHAPE> parser;
	marsh::Maps shaped;
	eigen::Packer<teq::Shape>().pack(shaped, expect);

	EXPECT_FATAL(parser(shaped, {}), eigen::no_argument_err.c_str());

	teq::Shape conflicting({3, 4, 5});
	EXPECT_FATAL(parser(shaped, {conflicting}),
		"cannot RESHAPE with shapes of different sizes 60 (shape [3\\4\\5\\1\\1\\1\\1\\1]) and 72 (shape [3\\4\\6\\1\\1\\1\\1\\1])");

	teq::Shape inshape({8, 3, 3});
	teq::Shape got = parser(shaped, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, Pad)
{
	eigen::PairVecT<teq::DimT> pads = {{3, 6}, {2, 3}, {0, 2}};
	egen::ShapeParser<egen::PAD> parser;
	marsh::Maps padded;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(padded, pads);

	EXPECT_FATAL(parser(padded, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({5, 4, 6, 7, 3});
	teq::Shape expect({14, 9, 8, 7, 3});
	teq::Shape got = parser(padded, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, Slice)
{
	eigen::PairVecT<teq::DimT> extents = {{3, 6}, {2, 3}, {0, 2}};
	egen::ShapeParser<egen::SLICE> parser;
	marsh::Maps exed;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(exed, extents);

	EXPECT_FATAL(parser(exed, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({5, 4, 6, 7, 3});
	teq::Shape expect({2, 2, 2, 7, 3});
	teq::Shape got = parser(exed, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, Stride)
{
	std::vector<teq::DimT> strides = {3, 2, 3};
	egen::ShapeParser<egen::STRIDE> parser;
	marsh::Maps strided;
	eigen::Packer<std::vector<teq::DimT>>().pack(strided, strides);

	EXPECT_FATAL(parser(strided, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({41, 4, 6, 7, 3});
	teq::Shape expect({14, 2, 2, 7, 3});
	teq::Shape got = parser(strided, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST(SHAPER, Scatter)
{
	teq::Shape expect({3, 4, 6});

	egen::ShapeParser<egen::SCATTER> parser;
	marsh::Maps shaped;
	eigen::Packer<teq::Shape>().pack(shaped, expect);

	EXPECT_FATAL(parser(shaped, {}), eigen::no_argument_err.c_str());

	// scatter allows conflicting shape
	teq::Shape conflicting({3, 4, 5});
	teq::Shape got = parser(shaped, {conflicting});

	EXPECT_ARREQ(expect, got);

	teq::Shape inshape({8, 3, 3});
	teq::Shape got2 = parser(shaped, {inshape});

	EXPECT_ARREQ(expect, got2);
}


TEST(SHAPER, Matmul)
{
	//
}


TEST(SHAPER, Conv)
{
	//
}


TEST(SHAPER, Concat)
{
	//
}


#endif // DISABLE_EIGEN_SHAPER_TEST
