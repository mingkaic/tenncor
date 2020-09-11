
#ifndef DISABLE_EIGEN_FUNCOPT_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mutable_leaf.hpp"
#include "internal/eigen/eigen.hpp"


TEST(FUNCOPT, Default)
{
	egen::FuncOpt<egen::SUB> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	auto b = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b}));
}


TEST(FUNCOPT, Reduce)
{
	egen::FuncOpt<egen::REDUCE_SUM> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

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
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

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
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

	eigen::Packer<std::vector<teq::RankT>> packer;

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
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

	eigen::Packer<std::vector<teq::DimT>> packer;

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
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

	eigen::Packer<teq::Shape> packer;

	packer.pack(attrs, teq::Shape({3, 2}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, outshape);
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Slice)
{
	egen::FuncOpt<egen::SLICE> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

	eigen::Packer<eigen::PairVecT<teq::DimT>> packer;

	// no slice coverage
	packer.pack(attrs, {});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{1, 2}, {4, 0}});
	EXPECT_FATAL(opt.operator()<double>(attrs, {a}),
		"cannot create slice with 0 dimensions (second value of extents) (extents=[1:2\\4:0])");
	attrs.rm_attr(packer.get_key());

	// slice coverage greater than shape
	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{0, 3}, {0, 4}});
	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	attrs.rm_attr(packer.get_key());

	packer.pack(attrs, eigen::PairVecT<teq::DimT>{{1, 2}, {1, 3}});
	EXPECT_FALSE(opt.operator()<double>(attrs, {a}));
}


TEST(FUNCOPT, Pad)
{
	egen::FuncOpt<egen::PAD> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

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
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	auto b = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	auto c = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);

	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b}));
	EXPECT_FALSE(opt.operator()<double>(attrs, {a, b, c}));
}


TEST(FUNCOPT, Cast)
{
	egen::FuncOpt<egen::CAST> opt;
	marsh::Maps attrs;

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	a->meta_.tcode_ = egen::DOUBLE;

	EXPECT_TRUE(opt.operator()<double>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<float>(attrs, {a}));
	EXPECT_FALSE(opt.operator()<int32_t>(attrs, {a}));
}


#endif // DISABLE_EIGEN_FUNCOPT_TEST
