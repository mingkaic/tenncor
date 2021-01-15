
#ifndef DISABLE_EIGEN_SHAPER_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/eigen/eigen.hpp"


using ::testing::_;
using ::testing::An;
using ::testing::Return;
using ::testing::Throw;


struct SHAPER : public tutil::TestcaseWithLogger<> {};


TEST_F(SHAPER, Default)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::ShapeParser<egen::ABS> parser;
	marsh::Maps empty;

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(empty, {}), eigen::no_argument_err.c_str());

	teq::ShapesT conflicting = {
		teq::Shape({3, 4, 5}),
		teq::Shape({3, 2, 5}),
	};

	std::string fatalmsg = "cannot ABS with incompatible shapes [3\\2\\5\\1\\1\\1\\1\\1] and [3\\4\\5\\1\\1\\1\\1\\1]";
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(empty, conflicting), fatalmsg.c_str());

	teq::Shape expect({3, 4, 6});
	teq::ShapesT good = {expect, expect};
	teq::Shape got = parser(empty, good);

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Identity)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::ShapeParser<egen::IDENTITY> parser;
	marsh::Maps empty;

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(empty, {}), eigen::no_argument_err.c_str());

	teq::Shape a({3, 4, 5});
	teq::Shape b({3, 2, 5});

	auto good = parser(empty, {a, b});
	EXPECT_ARREQ(a, good);
}


TEST_F(SHAPER, Reduce)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	std::set<teq::RankT> rdims = {3, 2, 5};

	egen::ShapeParser<egen::REDUCE_SUM> parser;
	marsh::Maps dimmed;
	eigen::Packer<std::set<teq::RankT>>().pack(dimmed, rdims);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(dimmed, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({3, 4, 6, 7, 3});
	teq::Shape expect({3, 4, 1, 1, 3});
	teq::Shape got = parser(dimmed, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, ArgReduce)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	teq::RankT rdim = 3;
	egen::ShapeParser<egen::ARGMAX> parser;
	marsh::Maps dimmed;
	eigen::Packer<teq::RankT>().pack(dimmed, rdim);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
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


TEST_F(SHAPER, Permute)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	teq::RanksT rdims = {2, 1, 0};

	egen::ShapeParser<egen::PERMUTE> parser;
	marsh::Maps dimmed;
	eigen::Packer<teq::RanksT>().pack(dimmed, rdims);

	teq::Shape inshape({3, 4, 6, 7, 3});
	teq::Shape expect({6, 4, 3, 7, 3});
	teq::Shape got = parser(dimmed, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Extend)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	egen::ShapeParser<egen::EXTEND> parser;
	marsh::Maps good;
	marsh::Maps bad;
	eigen::Packer<teq::DimsT>().pack(good, {1, 2, 1});
	eigen::Packer<teq::DimsT>().pack(bad, {1, 2, 0});

	teq::Shape goodshape({3, 1, 6, 7, 3});
	teq::Shape badshape({3, 4, 6, 7, 3});

	std::string fatalmsg = "cannot extend using zero dimensions [1\\2\\0]";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(bad, {goodshape}), fatalmsg.c_str());
	std::string fatalmsg1 = "cannot extend non-singular dimension 1 of shape [3\\4\\6\\7\\3\\1\\1\\1]: bcast=[1\\2\\1]";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(parser(good, {badshape}), fatalmsg1.c_str());

	teq::Shape expect({3, 2, 6, 7, 3});
	teq::Shape got = parser(good, {goodshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Reshape)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	teq::Shape expect({3, 4, 6});

	egen::ShapeParser<egen::RESHAPE> parser;
	marsh::Maps shaped;
	eigen::Packer<teq::Shape>().pack(shaped, expect);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(shaped, {}), eigen::no_argument_err.c_str());

	teq::Shape conflicting({3, 4, 5});
	std::string fatalmsg = "cannot RESHAPE with shapes of different sizes 60 (shape [3\\4\\5\\1\\1\\1\\1\\1]) and 72 (shape [3\\4\\6\\1\\1\\1\\1\\1])";
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(shaped, {conflicting}), fatalmsg.c_str());

	teq::Shape inshape({8, 3, 3});
	teq::Shape got = parser(shaped, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Pad)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	eigen::PairVecT<teq::DimT> pads = {{3, 6}, {2, 3}, {0, 2}};
	egen::ShapeParser<egen::PAD> parser;
	marsh::Maps padded;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(padded, pads);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(padded, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({5, 4, 6, 7, 3});
	teq::Shape expect({14, 9, 8, 7, 3});
	teq::Shape got = parser(padded, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Slice)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	eigen::PairVecT<teq::DimT> extents = {{3, 6}, {2, 3}, {0, 2}};
	egen::ShapeParser<egen::SLICE> parser;
	marsh::Maps exed;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(exed, extents);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(exed, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({5, 4, 6, 7, 3});
	teq::Shape expect({2, 2, 2, 7, 3});
	teq::Shape got = parser(exed, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Stride)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	teq::DimsT strides = {3, 2, 3};
	egen::ShapeParser<egen::STRIDE> parser;
	marsh::Maps strided;
	eigen::Packer<teq::DimsT>().pack(strided, strides);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(strided, {}), eigen::no_argument_err.c_str());

	teq::Shape inshape({41, 4, 6, 7, 3});
	teq::Shape expect({14, 2, 2, 7, 3});
	teq::Shape got = parser(strided, {inshape});

	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, Scatter)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	teq::Shape expect({3, 4, 6});

	egen::ShapeParser<egen::SCATTER> parser;
	marsh::Maps shaped;
	eigen::Packer<teq::Shape>().pack(shaped, expect);

	EXPECT_CALL(*logger_, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::fatal_level, eigen::no_argument_err, _)).Times(1).WillOnce(Throw(exam::TestException(eigen::no_argument_err)));
	EXPECT_FATAL(parser(shaped, {}), eigen::no_argument_err.c_str());

	// scatter allows conflicting shape
	teq::Shape conflicting({3, 4, 5});
	teq::Shape got = parser(shaped, {conflicting});

	EXPECT_ARREQ(expect, got);

	teq::Shape inshape({8, 3, 3});
	teq::Shape got2 = parser(shaped, {inshape});

	EXPECT_ARREQ(expect, got2);
}


TEST_F(SHAPER, Matmul)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	egen::ShapeParser<egen::CONTRACT> parser;
	eigen::Packer<eigen::PairVecT<teq::RankT>> packer;

	teq::Shape c({3, 4, 6});
	marsh::Maps same;
	packer.pack(same, {{0, 0}, {1, 1}, {2, 2}});
	teq::Shape wun = parser(same, {c, c});
	teq::Shape expect;
	EXPECT_ARREQ(expect, wun);

	teq::Shape sym({2, 2, 2});
	marsh::Maps bad;
	packer.pack(bad, {{0, 0}, {0, 1}, {2, 0}});
	std::string fatalmsg = "contraction dimensions [0:0\\0:1\\2:0] must be unique for each side";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(bad, {sym, sym}), fatalmsg.c_str());

	marsh::Maps outer;
	packer.pack(outer, {});
	teq::Shape dup = parser(outer, {c, c});
	teq::Shape expect1({3, 4, 6, 3, 4, 6});
	EXPECT_ARREQ(expect1, dup);

	teq::Shape a({4, 3, 6});
	teq::Shape b({3, 5, 6});
	teq::Shape a2d({4, 3});
	teq::Shape b2d({3, 5});
	marsh::Maps transposed;
	packer.pack(transposed, {{1, 0}});

	std::string fatalmsg1 = "invalid shapes [3\\4\\6\\1\\1\\1\\1\\1] and [3\\4\\6\\1\\1\\1\\1\\1] "
		"do not match common dimensions [1:0]";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(parser(transposed, {c, c}), fatalmsg1.c_str());

	teq::Shape trans3d = parser(transposed, {a, b});
	teq::Shape expect2({5, 6, 4, 6});
	EXPECT_ARREQ(expect2, trans3d);

	teq::Shape trans2d = parser(transposed, {a2d, b2d});
	teq::Shape expect3({5, 4});
	EXPECT_ARREQ(expect3, trans2d);

	teq::Shape a2({3, 4, 6});
	teq::Shape b2({5, 3, 6});
	teq::Shape a2d2({3, 4});
	teq::Shape b2d2({5, 3});
	marsh::Maps typical;
	packer.pack(typical, {{0, 1}});

	teq::Shape norm3d = parser(typical, {a2, b2});
	EXPECT_ARREQ(expect2, norm3d);

	teq::Shape norm2d = parser(typical, {a2d2, b2d2});
	EXPECT_ARREQ(expect3, norm2d);
}


TEST_F(SHAPER, Conv)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	egen::ShapeParser<egen::CONV> parser;
	eigen::Packer<teq::RanksT> packer;

	teq::Shape imgshape({4, 5, 6, 7});
	teq::Shape kernshape({3, 2, 5});

	marsh::Maps badranks;
	packer.pack(badranks, {0, 2});
	std::string fatalmsg = "cannot have ambiguous ranks not specified in kernelshape "
		"[3\\2\\5\\1\\1\\1\\1\\1] (ranks=[0\\2])";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg.c_str(), _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(badranks, {imgshape, kernshape}), fatalmsg.c_str());

	marsh::Maps badkern;
	packer.pack(badkern, {2, 1, 0});
	std::string fatalmsg1 = "cannot convolve a kernel of shape [3\\2\\5\\1\\1\\1\\1\\1] "
		"against smaller image of shape [4\\5\\6\\7\\1\\1\\1\\1] at "
		"dimensions (shape:kernel=0:2)";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(parser(badkern, {imgshape, kernshape}), fatalmsg1.c_str());

	marsh::Maps attrs;
	packer.pack(attrs, {2, 1, 3});
	auto got = parser(attrs, {imgshape, kernshape});
	teq::Shape expect({4, 4, 4, 3});
	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, ConcatBinary)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::ShapeParser<egen::CONCAT> parser;
	eigen::Packer<teq::RankT> packer;

	teq::Shape a({3, 4, 5, 6});
	teq::Shape b({3, 4, 2, 6});

	marsh::Maps badranks;
	packer.pack(badranks, 1);
	std::string fatalmsg = "cannot group concat incompatible shapes [3\\4\\5\\6\\1\\1\\1\\1] and "
		"[3\\4\\2\\6\\1\\1\\1\\1] along axis 1";
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(badranks, {a, b}), fatalmsg.c_str());

	marsh::Maps attrs;
	packer.pack(attrs, 2);
	auto got = parser(attrs, {a, b});
	teq::Shape expect({3, 4, 7, 6});
	EXPECT_ARREQ(expect, got);
}


TEST_F(SHAPER, ConcatNnary)
{
	EXPECT_CALL(*logger_, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	egen::ShapeParser<egen::CONCAT> parser;
	eigen::Packer<teq::RankT> packer;

	teq::Shape a({3, 4, 1, 6});
	teq::Shape b({3, 4, 4, 6});

	marsh::Maps badranks;
	packer.pack(badranks, 1);
	std::string fatalmsg = "cannot group concat incompatible shapes [3\\4\\1\\6\\1\\1\\1\\1] and "
		"[3\\4\\4\\6\\1\\1\\1\\1] along axis 1";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(parser(badranks, {a, a, b}), fatalmsg.c_str());

	marsh::Maps attrs;
	packer.pack(attrs, 2);

	std::string fatalmsg1 = "cannot group concat shapes with dimension that is not one";
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(parser(attrs, {a, b, a}), fatalmsg1.c_str());

	auto got = parser(attrs, {a, a, a});
	teq::Shape expect({3, 4, 3, 6});
	EXPECT_ARREQ(expect, got);
}


#endif // DISABLE_EIGEN_SHAPER_TEST
