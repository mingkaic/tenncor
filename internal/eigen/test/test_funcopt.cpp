
#ifndef DISABLE_EIGEN_FUNCOPT_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/global/global.hpp"

#include "internal/eigen/mock/mock.hpp"

#include "testutil/tutil.hpp"


using ::testing::_;
using ::testing::An;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Throw;


TEST(FUNCOPT, Default)
{
	egen::FuncOpt<egen::SUB> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);
	auto b = make_var(outshape);
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b}));
}


TEST(FUNCOPT, Reduce)
{
	egen::FuncOpt<egen::REDUCE_SUM> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<std::set<teq::RankT>> packer;
	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, {1});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	// ignore inshape with respect to significant dimensions
	packer.pack(attrs, {2});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, ArgReduce)
{
	egen::FuncOpt<egen::ARGMAX> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<teq::RankT> packer;

	packer.pack(attrs, 1);
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, 2);
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, teq::rank_cap);
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Permute)
{
	egen::FuncOpt<egen::PERMUTE> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<teq::RanksT> packer;

	// outshape completely different
	packer.pack(attrs, {1, 2, 0});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, {0, 2, 1});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	// outshape is same but operationally different
	packer.pack(attrs, {1, 0});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	// redundant
	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, {0, 1, 2});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Extend)
{
	egen::FuncOpt<egen::EXTEND> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<teq::DimsT> packer;

	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));

	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, {1, 1, 1});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, {1, 1, 2});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Reshape)
{
	egen::FuncOpt<egen::RESHAPE> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<teq::Shape> packer;

	packer.pack(attrs, teq::Shape({3, 2}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, outshape);
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Slice)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	egen::FuncOpt<egen::SLICE> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<eigen::PairVecT<teq::DimT>> packer;

	// no slice coverage
	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{1, 2}, {4, 0}});
	std::string fatalmsg = "cannot create slice with 0 dimensions (second value of extents) (extents=[1:2\\4:0])";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));;
	EXPECT_FATAL(opt.operator()<double>(attrs, {a}), fatalmsg.c_str());
	attrs.rm_attr(packer.get_key());

	// slice coverage greater than shape
	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{0, 3}, {0, 4}});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{1, 2}, {1, 3}});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));

	global::set_logger(new exam::NoSupportLogger());
}


TEST(FUNCOPT, Pad)
{
	egen::FuncOpt<egen::PAD> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);

	eigen::Packer<eigen::PairVecT<teq::DimT>> packer;

	// no padding
	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	// zero padding
	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{0, 0}, {0, 0}});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{0, 3}, {4, 0}});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());
}


TEST(FUNCOPT, Add)
{
	egen::FuncOpt<egen::ADD> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);
	auto b = make_var(outshape);
	auto c = make_var(outshape);

	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b, c}));
}


TEST(FUNCOPT, Cast)
{
	egen::FuncOpt<egen::CAST> opt;
	marsh::Maps attrs;

	MockMeta mockmeta;
	teq::Shape outshape({2, 2});
	auto a = make_var(outshape);
	EXPECT_CALL(*a, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<float>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<int32_t>(attrs, {a}));
}


#endif // DISABLE_EIGEN_FUNCOPT_TEST
