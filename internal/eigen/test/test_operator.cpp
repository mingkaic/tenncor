
#ifndef DISABLE_EIGEN_OPERATOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mutable_leaf.hpp"
#include "internal/eigen/eigen.hpp"


static void test_reduce (
	std::function<eigen::EigenptrT(teq::Shape,
		const teq::iTensor&,const marsh::Maps& attr)> red,
	std::function<double(double,double)> agg)
{
	std::set<teq::RankT> rranks = {1};
	marsh::Maps mvalues;
	eigen::Packer<std::set<teq::RankT>>().pack(mvalues, rranks);

	MockLeaf edge(std::vector<double>{2, 3, 4, 5, 6, 7}, teq::Shape({3, 2}));
	auto r = red(teq::Shape({3}), edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();
	EXPECT_EQ(agg(2, 5), raw[0]);
	EXPECT_EQ(agg(3, 6), raw[1]);
	EXPECT_EQ(agg(4, 7), raw[2]);
}


TEST(OPERATOR, ReduceSum)
{
	test_reduce(eigen::reduce_sum<double>, [](double a, double b) { return a + b; });
}


TEST(OPERATOR, ReduceProd)
{
	test_reduce(eigen::reduce_prod<double>, [](double a, double b) { return a * b; });
}


TEST(OPERATOR, ReduceMin)
{
	test_reduce(eigen::reduce_min<double>, [](double a, double b) { return std::min(a, b); });
}


TEST(OPERATOR, ReduceMax)
{
	test_reduce(eigen::reduce_max<double>, [](double a, double b) { return std::max(a, b); });
}


TEST(OPERATOR, ArgMax)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 1);

	MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({3, 2}));
	auto r = eigen::argmax<double>(teq::Shape({3}), edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();
	EXPECT_EQ(1, raw[0]);
	EXPECT_EQ(0, raw[1]);
	EXPECT_EQ(1, raw[2]);

	marsh::Maps mvalues2;
	eigen::Packer<teq::RankT>().pack(mvalues2, 8);

	MockLeaf edge2(std::vector<double>{2, 8, 4, 5, 9, 7}, teq::Shape({3, 2}));
	auto r2 = eigen::argmax<double>(teq::Shape({1}), edge2, mvalues2);

	double* raw2 = (double*) r2->data();
	r2->assign();
	EXPECT_EQ(4, raw2[0]);
}


TEST(OPERATOR, Extend)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {1, 4});

	teq::Shape outshape({3, 4, 2});
	MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({3, 1, 2}));
	auto r = eigen::extend<double>(outshape, edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 2, 8, 4, 2, 8, 4, 2, 8, 4,
		5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Permute)
{
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{2, 0, 1, 3, 4, 5, 6, 7});

		teq::Shape outshape({3, 2, 2});
		MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12}, teq::Shape({2, 2, 3}));
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{ // same thing as above block except exclude 6 and 7 values
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{2, 0, 1, 3, 4, 5});

		teq::Shape outshape({3, 2, 2});
		MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12}, teq::Shape({2, 2, 3}));
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{1, 0, 2, 3, 4, 5, 6, 7});

		teq::Shape outshape({3, 2});
		MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({2, 3}));
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 4, 6, 8, 5, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Reshape)
{
	teq::Shape outshape({1, 3, 2});
	MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({2, 1, 3}));
	auto r = eigen::reshape<double>(outshape, edge);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Slice)
{
	MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({3, 2}));
	// slice both dimensions 0 and 1
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{1, 2}, {1, 1}});

		teq::Shape outshape({2, 1});
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	// slice last dimension (validate optimization)
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{0, 3}, {1, 1}});

		teq::Shape outshape({3, 1});
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {5, 6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, MultiConcat)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	{
		teq::Shape outshape({2, 4});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5}, teq::Shape({1, 4}));
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9}, teq::Shape({1, 4}));
		auto r = eigen::concat<double>(outshape, teq::TensptrsT{
			edgea, edgeb}, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {
			2, 1,
			8, 0,
			4, 3,
			5, 9,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({3, 4});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5}, teq::Shape({1, 4}));
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9}, teq::Shape({1, 4}));
		auto edgec = std::make_shared<MockLeaf>(
			std::vector<double>{3, 7, 2, 11}, teq::Shape({1, 4}));
		auto r = eigen::concat<double>(outshape, teq::TensptrsT{
			edgea, edgeb, edgec}, mvalues);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {
			2, 1, 3,
			8, 0, 7,
			4, 3, 2,
			5, 9, 11,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Pow)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 3, 2, 4}, outshape);
		auto r = eigen::pow<double>(outshape, *edgea, *edgeb);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 1, 64, 125, 36, 2401};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 4, 2}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 3, 2, 4, 2, 3}, outshape);
		auto r = eigen::pow<double>(outshape, *edgea, *edgeb);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 1, 64, 125, 36, 2401, 16, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Add)
{
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 8, 11}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9, 10, 11, 6, 1.2}, outshape);

		auto r = eigen::add<double>(outshape, teq::TensptrsT{
			edgea, edgeb});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18, 14, 12.2};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	teq::Shape outshape({2, 3});
	auto edgea = std::make_shared<MockLeaf>(
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	auto edgeb = std::make_shared<MockLeaf>(
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	auto edgec = std::make_shared<MockLeaf>(
		std::vector<double>{4.2, 1, 7.1, 1, 2, 1.1}, outshape);

	{
		auto r = eigen::add<double>(outshape, teq::TensptrsT{
			edgea, edgeb});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		auto r = eigen::add<double>(outshape, teq::TensptrsT{
			edgea, edgeb, edgec});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {7.2, 9, 14.1, 15, 18, 19.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Sub)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);

		auto r = eigen::sub<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 8, 11}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9, 10, 11, 6, 1.2}, outshape);

		auto r = eigen::sub<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4, 2, 9.8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Mul)
{
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 1.2, 3}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0, 3, 9, 10, 11, 2, 1.7}, outshape);

		auto r = eigen::mul<double>(outshape, teq::TensptrsT{
			edgea, edgeb});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77, 2.4, 5.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}

	teq::Shape outshape({2, 3});
	auto edgea = std::make_shared<MockLeaf>(
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	auto edgeb = std::make_shared<MockLeaf>(
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	auto edgec = std::make_shared<MockLeaf>(
		std::vector<double>{4, 1, 7, 1, 2, 1}, outshape);

	{
		auto r = eigen::mul<double>(outshape, teq::TensptrsT{
			edgea, edgeb});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		auto r = eigen::mul<double>(outshape, teq::TensptrsT{
			edgea, edgeb, edgec});
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {8, 0, 84, 45, 120, 77};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Div)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 3, 9, 10, 11}, outshape);

		auto r = eigen::div<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 16, 4./3, 5./9, 0.6, 7./11};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 1.2, 3}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 3, 9, 10, 11, 2, 1.7}, outshape);

		auto r = eigen::div<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 16, 4./3, 5./9, 0.6, 7./11, 0.6, 3/1.7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Eq)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::eq<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {0, 0, 1, 0, 1, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::eq<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {0, 0, 1, 0, 1, 0, 1, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Neq)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::neq<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 1, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::neq<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 1, 0, 1, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Lt)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::lt<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {0, 0, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::lt<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {0, 0, 0, 1, 0, 1, 0, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Gt)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::gt<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 1, 0, 0, 0, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::gt<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 1, 0, 0, 0, 0, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Min)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::min<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 0.5, 4, 5, 6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::min<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 0.5, 4, 5, 6, 7, 3, 3};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Max)
{
	{
		teq::Shape outshape({2, 3});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::max<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 8, 4, 9, 6, 11};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 4, 5, 6, 7, 3, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::max<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {2, 8, 4, 9, 6, 11, 3, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, RandUniform)
{
	{
		teq::Shape outshape({2, 3});
		std::vector<double> a{1, 0.5, 3.5, 5, 6, 7};
		std::vector<double> b{2, 8, 4, 9, 6, 11};
		auto edgea = std::make_shared<MockLeaf>(a, outshape);
		auto edgeb = std::make_shared<MockLeaf>(b, outshape);

		auto r = eigen::rand_uniform<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		for (size_t i = 0, n = got_raw.size(); i < n; ++i)
		{
			double e = got_raw[i];
			EXPECT_LE(a[i], e);
			EXPECT_GE(b[i], e);
		}
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> a{1, 0.5, 3.5, 5, 6, 7, 3, 8};
		std::vector<double> b{2, 0.5, 4, 9, 6, 11, 4, 13};
		auto edgea = std::make_shared<MockLeaf>(a, outshape);
		auto edgeb = std::make_shared<MockLeaf>(b, outshape);

		auto r = eigen::rand_uniform<double>(outshape, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		for (size_t i = 0, n = got_raw.size(); i < n; ++i)
		{
			double e = got_raw[i];
			EXPECT_LE(a[i], e);
			EXPECT_GE(b[i], e);
		}
	}
}


TEST(OPERATOR, Select)
{
	{
		teq::Shape outshape({2, 3});
		auto comp = std::make_shared<MockLeaf>(
			std::vector<double>{0, 1, 0, 0, 1, 1}, outshape);
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 9, 5, 8, 7}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11}, outshape);

		auto r = eigen::select<double>(outshape,
			*comp, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 8, 4, 9, 8, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		auto comp = std::make_shared<MockLeaf>(
			std::vector<double>{0, 1, 0, 0, 1, 1, 0, 1}, outshape);
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{2, 8, 9, 5, 8, 7, 4, 8}, outshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{1, 0.5, 4, 9, 6, 11, 3, 3}, outshape);

		auto r = eigen::select<double>(outshape,
			*comp, *edgea, *edgeb);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {1, 8, 4, 9, 8, 7, 3, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Matmul)
{
	{
		teq::Shape outshape({2, 3});
		teq::Shape lshape({4, 3});
		teq::Shape rshape({2, 4});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{
				2, 8, 9, 5, 8, 7,
				1, 9, 4.2, 3, 2, 6,
			}, lshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{
				1, 0.5, 4, 9,
				6, 11, 3, 8,
			}, rshape);

		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().pack(mvalues, {{0, 1}});
		auto r = eigen::matmul<double>(outshape, *edgea, *edgeb, mvalues);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {103, 212, 69, 150, 46.2, 99.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 3});
		teq::Shape lshape({4, 2, 3});
		teq::Shape rshape({2, 4, 2});
		auto edgea = std::make_shared<MockLeaf>(
			std::vector<double>{
				2, 8, 9, 5, 8, 7,
				1, 9, 4.2, 3, 2, 6,
				2, 8, 9, 5, 8, 7,
				1, 9, 4.2, 3, 2, 6,
			}, lshape);
		auto edgeb = std::make_shared<MockLeaf>(
			std::vector<double>{
				1, 0.5, 4, 9,
				6, 11, 3, 8,
				1, 0.5, 4, 9,
				6, 11, 3, 8,
			}, rshape);

		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().pack(mvalues,
			{{0, 1}, {1, 2}});
		auto r = eigen::matmul<double>(outshape, *edgea, *edgeb, mvalues);
		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> expect_raw = {172, 362, 149.2, 311.1, 115.2, 249.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Pad)
{
	marsh::Maps mvalues;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
		{{1, 1}});

	teq::Shape outshape({4, 3});
	MockLeaf edge(
		std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({2, 3}));
	auto r = eigen::pad<double>(outshape, edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		0, 2, 8, 0,
		0, 4, 5, 0,
		0, 6, 7, 0,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Stride)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {1, 2});

	teq::Shape outshape({2, 2});
	MockLeaf edge(
		std::vector<double>{2, 8, 4, 5, 6, 7}, teq::Shape({2, 3}));
	auto r = eigen::stride<double>(outshape, edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Scatter)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {2, 2});

	teq::Shape outshape({3, 3});
	MockLeaf edge(
		std::vector<double>{2, 8, 4, 5}, teq::Shape({2, 2}));
	auto r = eigen::scatter<double>(outshape, edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2, 0, 8,
		0, 0, 0,
		4, 0, 5,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Reverse)
{
	marsh::Maps mvalues;
	eigen::Packer<std::set<teq::RankT>>().pack(mvalues, {1});

	teq::Shape outshape({2, 3});
	MockLeaf edge(std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	auto r = eigen::reverse<double>(outshape, edge, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		6, 7, 4, 5, 2, 8
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Concat)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);

	teq::Shape outshape({3, 3});
	auto edgea = std::make_shared<MockLeaf>(
		std::vector<double>{2, 8, 4, 5, 7, 6}, teq::Shape({2, 3}));
	auto edgeb = std::make_shared<MockLeaf>(
		std::vector<double>{1, 0, 3}, teq::Shape({1, 3}));
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edgea, edgeb}, mvalues);

	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 1,
		4, 5, 0,
		7, 6, 3,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
}


#define _DBLARRCHECK(ARR, ARR2, GBOOL) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	GBOOL(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }
#define ASSERT_ARRDBLEQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	ASSERT_EQ(ARR.size(), ARR2.size()) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead";\
	for (size_t i = 0, n = ARR.size(); i < n; ++i){\
		ASSERT_DOUBLE_EQ(ARR[i], ARR2[i]) <<\
			"expect list " << arrs.str() << ", got " << arrs2.str() << " instead";\
	} }


static void elementary_unary (
	std::function<eigen::EigenptrT(teq::Shape,const teq::iTensor&)> f,
	std::function<double(double)> unary,
	std::vector<double> invec = {-2, 8, -4, -5, 7, 6})
{
	std::vector<double> expect;
	std::transform(invec.begin(), invec.end(), std::back_inserter(expect),
		[&](double e) { return unary(e); });
	{
		teq::Shape shape({2, 3});
		MockLeaf edge(invec, shape);
		auto r = f(shape, edge);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
	}
	{
		teq::Shape shape({2, 1, 3});
		MockLeaf edge(invec, shape);
		auto r = f(shape, edge);

		double* raw = (double*) r->data();
		r->assign();

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
	}
}


TEST(OPERATOR, Abs)
{
	elementary_unary(eigen::abs<double>, [](double e) { return std::abs(e); });
}


TEST(OPERATOR, Neg)
{
	elementary_unary(eigen::neg<double>, [](double e) { return -e; });
}


TEST(OPERATOR, Sin)
{
	elementary_unary(eigen::sin<double>, [](double e) { return std::sin(e); });
}


TEST(OPERATOR, Cos)
{
	elementary_unary(eigen::cos<double>, [](double e) { return std::cos(e); });
}


TEST(OPERATOR, Tan)
{
	elementary_unary(eigen::tan<double>, [](double e) { return std::tan(e); });
}


TEST(OPERATOR, Exp)
{
	elementary_unary(eigen::exp<double>, [](double e) { return std::exp(e); });
}


TEST(OPERATOR, Log)
{
	elementary_unary(eigen::log<double>, [](double e) { return std::log(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Sqrt)
{
	elementary_unary(eigen::sqrt<double>, [](double e) { return std::sqrt(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Round)
{
	elementary_unary(eigen::round<double>, [](double e) { return std::round(e); },
		{3.22, 8.51, 2.499, 5.2, 7.17, 3.79});
}


TEST(OPERATOR, Sigmoid)
{
	elementary_unary(
		[](teq::Shape outshape, const teq::iTensor& in)
		{ return eigen::sigmoid<double>(outshape, in); },
		[](double e) { return 1. / (1. + std::exp(-e)); });
}


TEST(OPERATOR, Tanh)
{
	elementary_unary(
		[](teq::Shape outshape, const teq::iTensor& in)
		{ return eigen::tanh<double>(outshape, in); },
		[](double e) { return std::tanh(e); });
}


TEST(OPERATOR, Square)
{
	elementary_unary(eigen::square<double>,
		[](double e) { return e * e; });
}


TEST(OPERATOR, Cube)
{
	elementary_unary(
		[](teq::Shape outshape, const teq::iTensor& in)
		{ return eigen::cube<double>(outshape, in); },
		[](double e) { return e * e * e; });
}


TEST(OPERATOR, Convolution)
{
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues, {1, 1});
		teq::Shape outshape({3, 2});
		MockLeaf image(std::vector<double>{
			2, 8, 4,
			5, 7, 6,
			9, 1, 0,
		}, teq::Shape({3, 3}));
		MockLeaf kernel(std::vector<double>{0.3, 0.6}, teq::Shape({2}));

		EXPECT_FATAL(eigen::convolution<double>(outshape, image, kernel, mvalues),
			"convolution does not support repeated kernel dimensions: [1\\1]");
	}

	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues, {1});
		teq::Shape outshape({3, 2});
		MockLeaf image(std::vector<double>{
			2, 8, 4,
			5, 7, 6,
			9, 1, 0,
		}, teq::Shape({3, 3}));
		MockLeaf kernel(std::vector<double>{0.3, 0.6, 4.0, 2.2}, teq::Shape({2, 2}));

		EXPECT_FATAL(eigen::convolution<double>(outshape, image, kernel, mvalues),
			"given kernel shape [2\\2\\1\\1\\1\\1\\1\\1], unspecified non-singular kernel dimension 1 is undefined");
	}

	marsh::Maps mvalues;
	eigen::Packer<teq::RanksT>().pack(mvalues, {1});

	teq::Shape outshape({3, 2});
	MockLeaf image(std::vector<double>{
		2, 8, 4,
		5, 7, 6,
		9, 1, 0,
	}, teq::Shape({3, 3}));
	MockLeaf kernel(std::vector<double>{0.3, 0.6}, teq::Shape({2}));

	auto r = eigen::convolution<double>(outshape, image, kernel, mvalues);
	double* raw = (double*) r->data();
	r->assign();

	std::vector<double> expect_raw = {
		2 * 0.3 + 5 * 0.6, 8 * 0.3 + 7 * 0.6, 4 * 0.3 + 6 * 0.6,
		5 * 0.3 + 9 * 0.6, 7 * 0.3 + 1 * 0.6, 6 * 0.3 + 0 * 0.6,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Assign)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	auto edgea = std::make_shared<MockMutableLeaf>(a, outshape);
	auto edgeb = std::make_shared<MockLeaf>(b, outshape);

	auto r = eigen::assign<double>(*edgea, *edgeb);
	double* raw = (double*) r->data();
	EXPECT_EQ(raw, edgea->device().data());
	r->assign();

	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(b, got_raw);
}


TEST(OPERATOR, AssignAdd)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	auto edgea = std::make_shared<MockMutableLeaf>(a, outshape);
	auto edgeb = std::make_shared<MockLeaf>(b, outshape);

	auto r = eigen::assign_add<double>(*edgea, *edgeb);
	double* raw = (double*) r->data();
	EXPECT_EQ(raw, edgea->device().data());
	r->assign();

	std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignSub)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	auto edgea = std::make_shared<MockMutableLeaf>(a, outshape);
	auto edgeb = std::make_shared<MockLeaf>(b, outshape);

	auto r = eigen::assign_sub<double>(*edgea, *edgeb);
	double* raw = (double*) r->data();
	EXPECT_EQ(raw, edgea->device().data());
	r->assign();

	std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignMul)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	auto edgea = std::make_shared<MockMutableLeaf>(a, outshape);
	auto edgeb = std::make_shared<MockLeaf>(b, outshape);

	auto r = eigen::assign_mul<double>(*edgea, *edgeb);
	double* raw = (double*) r->data();
	EXPECT_EQ(raw, edgea->device().data());
	r->assign();

	std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignDiv)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 2, 3, 9, 10, 11};
	auto edgea = std::make_shared<MockMutableLeaf>(a, outshape);
	auto edgeb = std::make_shared<MockLeaf>(b, outshape);

	auto r = eigen::assign_div<double>(*edgea, *edgeb);
	double* raw = (double*) r->data();
	EXPECT_EQ(raw, edgea->device().data());
	r->assign();

	std::vector<double> expect_raw = {2, 4, 4./3, 5./9, 0.6, 7./11};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Cast)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2.1, 8.5, 4.3, 5.2, 6.1, 7.2};
	auto edgea = std::make_shared<MockLeaf>(a, outshape);
	edgea->meta_.tcode_ = egen::DOUBLE;
	edgea->meta_.tname_ = egen::name_type(egen::DOUBLE);

	{
		auto r = eigen::cast<double>(*edgea);
		double* raw = (double*) r->data();
		EXPECT_EQ(raw, edgea->device().data());
		r->assign();

		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		ASSERT_ARRDBLEQ(a, got_raw);
	}
	{
		auto r = eigen::cast<int32_t>(*edgea);
		int32_t* raw = (int32_t*) r->data();
		r->assign();

		std::vector<int32_t> expect_raw = {2, 8, 4, 5, 6, 7};
		std::vector<int32_t> got_raw(raw, raw + outshape.n_elems());
		ASSERT_ARREQ(expect_raw, got_raw);
	}
}


#endif // DISABLE_EIGEN_OPERATOR_TEST
