
#ifndef DISABLE_EIGEN_OPERATOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/global/global.hpp"

#include "internal/eigen/mock/mock.hpp"

#include "testutil/tutil.hpp"


using ::testing::_;
using ::testing::An;
using ::testing::Const;
using ::testing::Invoke;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Throw;


template <typename T>
static MockLeafptrT make_sparse_var (std::vector<T>& dense_data,
	std::vector<eigen::StorageIdxT>& inner_indices,
	std::vector<eigen::StorageIdxT>& outer_indices,
	MockEigen& devref, const teq::Shape& shape,
	const std::string& label = "", jobs::GuardOpF once_exec = jobs::GuardOpF())
{
	eigen::SMatrixT<T> sm = eigen::make_matmap<T>(dense_data.data(), shape).sparseView();

	auto nzs = sm.nonZeros();
	auto sparse_data = sm.valuePtr();
	auto inidx = sm.innerIndexPtr();
	auto outidx = sm.outerIndexPtr();
	dense_data = std::vector<T>(sparse_data, sparse_data + nzs);
	inner_indices = std::vector<eigen::StorageIdxT>(inidx, inidx + nzs);
	// +1 for the last outer entry denoting end of indices
	outer_indices = std::vector<eigen::StorageIdxT>(outidx, outidx + shape.at(1) + 1);

	auto var = make_var(dense_data.data(), devref, shape, label, once_exec);
	EXPECT_CALL(devref, sparse_info()).WillRepeatedly(Return(eigen::SparseInfo{
		inner_indices.data(), outer_indices.data(), nzs}));
	return var;
}


template <typename T>
void expect_sparse_eq (const std::vector<T>& exdense_data,
	const teq::Shape& exshape, eigen::EigenptrT got)
{
	eigen::SMatrixT<T> sm = eigen::make_matmap<T>(
		(T*) exdense_data.data(), exshape).sparseView();
	T* exraw = sm.valuePtr();
	T* raw = (T*) got->data();
	ASSERT_NE(nullptr, raw);

	auto sinfo = got->sparse_info();
	ASSERT_TRUE(sinfo);
	ASSERT_EQ(sm.nonZeros(), sinfo->non_zeros_);

	std::vector<double> expect_raw(exraw, exraw + sm.nonZeros());
	std::vector<double> got_raw(raw, raw + sm.nonZeros());
	EXPECT_VECEQ(expect_raw, got_raw);

	auto eip = sm.innerIndexPtr();
	auto eop = sm.outerIndexPtr();
	std::vector<eigen::StorageIdxT> expect_inner(eip, eip + sm.nonZeros());
	std::vector<eigen::StorageIdxT> expect_outer(eop, eop + exshape.at(1) + 1);
	std::vector<eigen::StorageIdxT> got_inner(
		sinfo->inner_indices_, sinfo->inner_indices_ + sm.nonZeros());
	std::vector<eigen::StorageIdxT> got_outer(
		sinfo->outer_indices_, sinfo->outer_indices_ + exshape.at(1) + 1);
	EXPECT_VECEQ(expect_inner, got_inner);
	EXPECT_VECEQ(expect_outer, got_outer);
}


static void test_reduce (
	std::function<eigen::EigenptrT(teq::Shape,
		const teq::TensptrT&,const marsh::Maps& attr)> red,
	std::function<double(double,double)> agg)
{
	teq::RanksT ranks = {0, 1};
	for (teq::RankT rank : ranks)
	{
		std::set<teq::RankT> rranks = {rank};
		teq::Shape inshape({3, 2});
		teq::DimT outdim;
		teq::Shape outshape;
		if (rank == 0)
		{
			outdim = 2;
			outshape = {1, 2};
		}
		else
		{
			outdim = 3;
			outshape = {3};
		}
		marsh::Maps mvalues;
		eigen::Packer<std::set<teq::RankT>>().pack(mvalues, rranks);
		std::vector<double> outdata(outdim);
		auto memory = std::make_shared<MockRuntimeMemory>();

		{
			size_t lifetimes = 0;
			auto incr_life = [&lifetimes]{ ++lifetimes; };

			std::vector<double> expect_raw{2, 3, 4, 5, 6, 7};
			MockDeviceRef mockdev;
			auto edge = make_var(expect_raw.data(), mockdev, inshape, "", incr_life);

#ifndef PERM_OP
			auto outbytes = outdim * sizeof(double);
			EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
			EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
			auto r = red(outshape, edge, mvalues);

			auto before = r->data();
			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			double* raw = (double*) r->data();
			ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
			EXPECT_EQ(nullptr, before);
			EXPECT_EQ(outdata.data(), raw);
#endif

			if (rank == 1)
			{
				EXPECT_EQ(agg(2, 5), raw[0]);
				EXPECT_EQ(agg(3, 6), raw[1]);
				EXPECT_EQ(agg(4, 7), raw[2]);
			}
			else
			{
				EXPECT_EQ(agg(agg(2, 3), 4), raw[0]);
				EXPECT_EQ(agg(agg(5, 6), 7), raw[1]);
			}
			EXPECT_EQ(1, lifetimes);
		}
	}
}


static void test_sparse_reduce (
	std::function<eigen::EigenptrT(teq::Shape,
		const teq::TensptrT&,const marsh::Maps& attr)> red,
	std::function<double(double,double)> agg)
{
	teq::RanksT ranks = {0, 1};
	for (teq::RankT rank : ranks)
	{
		std::set<teq::RankT> rranks = {rank};
		teq::Shape inshape({3, 2});
		teq::DimT outdim;
		teq::Shape outshape;
		if (rank == 0)
		{
			outdim = 2;
			outshape = {1, 2};
		}
		else
		{
			outdim = 3;
			outshape = {3};
		}
		marsh::Maps mvalues;
		eigen::Packer<std::set<teq::RankT>>().pack(mvalues, rranks);
		std::vector<double> outdata(outdim);
		auto memory = std::make_shared<MockRuntimeMemory>();

		{
			size_t lifetimes = 0;
			auto incr_life = [&lifetimes]{ ++lifetimes; };

			std::vector<double> expect_raw = {
				0, 2, 0,
				0, 4, 6,
			};
			std::vector<eigen::StorageIdxT> inner_indices;
			std::vector<eigen::StorageIdxT> outer_indices;
			MockEigen mockdev;
			auto edge = make_sparse_var(expect_raw,
				inner_indices, outer_indices,
				mockdev, inshape, "", incr_life);

#ifndef PERM_OP
			auto outbytes = outdim * sizeof(double);
			EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
			EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
			auto r = red(outshape, edge, mvalues);

			auto before = r->data();
			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			double* raw = (double*) r->data();
			ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
			EXPECT_EQ(nullptr, before);
			EXPECT_EQ(outdata.data(), raw);
#endif

			if (rank == 1)
			{
				EXPECT_EQ(agg(0, 0), raw[0]);
				EXPECT_EQ(agg(2, 4), raw[1]);
				EXPECT_EQ(agg(0, 6), raw[2]);
			}
			else
			{
				EXPECT_EQ(agg(agg(0, 2), 0), raw[0]);
				EXPECT_EQ(agg(agg(0, 4), 6), raw[1]);
			}
			EXPECT_EQ(1, lifetimes);
		}
	}
}


TEST(OPERATOR, Ref)
{
	std::vector<double> expect_raw = {2, 8, 4, 5, 6, 7};
	std::vector<double> expect_raw2 = expect_raw;
	auto odata = expect_raw.data();
	teq::Shape outshape({2, 3});
	MockDeviceRef mockdev;
	auto origin = make_var(expect_raw.data(), mockdev, outshape);

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::ref(origin);

	auto raw = (double*) r->data();
	EXPECT_EQ(odata, raw);
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw2, got_raw);

	// do nothing
	eigen::RTMemptrT mem = memory;
	r->assign(0, mem);

	EXPECT_EQ(odata, raw);
	got_raw = std::vector<double>(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw2, got_raw);
}


TEST(OPERATOR, SparseRef)
{
	std::vector<double> expect_raw = {0, 8, 0, 0, 6, 7};
	std::vector<eigen::StorageIdxT> inner_indices;
	std::vector<eigen::StorageIdxT> outer_indices;
	MockEigen mockdev;
	teq::Shape outshape({2, 3});
	auto origin = make_sparse_var(expect_raw, inner_indices,
		outer_indices, mockdev, outshape);
	std::vector<double> expect_raw2 = expect_raw;

	auto odata = expect_raw.data();
	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::ref(origin);

	auto raw = (double*) r->data();
	EXPECT_EQ(odata, raw);
	auto sinfo = r->sparse_info();
	ASSERT_TRUE(sinfo);
	ASSERT_EQ(3, sinfo->non_zeros_);
	std::vector<double> got_raw(raw, raw + expect_raw.size());
	std::vector<eigen::StorageIdxT> got_inner(
		sinfo->inner_indices_, sinfo->inner_indices_ + 3);
	std::vector<eigen::StorageIdxT> got_outer(
		sinfo->outer_indices_, sinfo->outer_indices_ + 4);
	EXPECT_VECEQ(expect_raw2, got_raw);
	EXPECT_VECEQ(inner_indices, got_inner);
	EXPECT_VECEQ(outer_indices, got_outer);

	// do nothing
	eigen::RTMemptrT mem = memory;
	r->assign(0, mem);

	EXPECT_EQ(odata, raw);
	sinfo = r->sparse_info();
	ASSERT_TRUE(sinfo);
	ASSERT_EQ(expect_raw.size(), sinfo->non_zeros_);
	got_raw = std::vector<double>(raw, raw + 3);
	got_inner = std::vector<eigen::StorageIdxT>(
		sinfo->inner_indices_, sinfo->inner_indices_ + 3);
	got_outer = std::vector<eigen::StorageIdxT>(
		sinfo->outer_indices_, sinfo->outer_indices_ + 4);
	EXPECT_VECEQ(expect_raw2, got_raw);
	EXPECT_VECEQ(inner_indices, got_inner);
	EXPECT_VECEQ(outer_indices, got_outer);
}


TEST(OPERATOR, ReduceSum)
{
	test_reduce(eigen::reduce_sum<double>,
	[](double a, double b){ return a + b; });
}


TEST(OPERATOR, SparseReduceSum)
{
	test_sparse_reduce(eigen::reduce_sum<double>,
	[](double a, double b){ return a + b; });
}


TEST(OPERATOR, ReduceProd)
{
	test_reduce(eigen::reduce_prod<double>,
	[](double a, double b){ return a * b; });
}


TEST(OPERATOR, SparseReduceProd)
{
	test_sparse_reduce(eigen::reduce_prod<double>,
	[](double a, double b){ return a * b; });
}


TEST(OPERATOR, ReduceMin)
{
	test_reduce(eigen::reduce_min<double>,
	[](double a, double b){ return std::min(a, b); });
}


TEST(OPERATOR, SparseReduceMin)
{
	test_sparse_reduce(eigen::reduce_min<double>,
	[](double a, double b){ return std::min(a, b); });
}


TEST(OPERATOR, ReduceMax)
{
	test_reduce(eigen::reduce_max<double>,
	[](double a, double b){ return std::max(a, b); });
}


TEST(OPERATOR, SparseReduceMax)
{
	test_sparse_reduce(eigen::reduce_max<double>,
	[](double a, double b){ return std::max(a, b); });
}


TEST(OPERATOR, ArgMax)
{
	auto memory = std::make_shared<MockRuntimeMemory>();
	size_t lifetimes = 0;
	auto incr_life = [&lifetimes]{ ++lifetimes; };

	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	teq::RanksT ranks = {0, 1};
	for (teq::RankT rank : ranks)
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RankT>().pack(mvalues, rank);

		MockDeviceRef mockdev;
		auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({3, 2}), "", incr_life);

		teq::DimT outdim;
		teq::Shape outshape;
		if (rank == 0)
		{
			outdim = 2;
			outshape = {1, 2};
		}
		else
		{
			outdim = 3;
			outshape = {3};
		}
		std::vector<double> outdata(6);

		{
#ifndef PERM_OP
			auto outbytes = outdim * sizeof(double);
			EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
			EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
			auto r = eigen::argmax<double>(outshape, edge, mvalues);

			auto before = r->data();
			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			double* raw = (double*) r->data();
			ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
			EXPECT_EQ(nullptr, before);
			EXPECT_EQ(outdata.data(), raw);
#endif

			if (0 == rank)
			{
				EXPECT_EQ(1, raw[0]);
				EXPECT_EQ(2, raw[1]);
			}
			else
			{
				EXPECT_EQ(1, raw[0]);
				EXPECT_EQ(0, raw[1]);
				EXPECT_EQ(1, raw[2]);
			}
			EXPECT_EQ(1, lifetimes);
		}
		lifetimes = 0;
	}

	marsh::Maps mvalues2;
	eigen::Packer<teq::RankT>().pack(mvalues2, 8);

	std::vector<double> orig_raw2{2, 8, 4, 5, 9, 7};
	MockDeviceRef mockdev2;
	auto edge2 = make_var(orig_raw2.data(), mockdev2, teq::Shape({3, 2}), "", incr_life);

	std::vector<double> outdata2(6);
	{
#ifndef PERM_OP
		auto outbytes2 = sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes2)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes2)).Times(1);
#endif
		auto r2 = eigen::argmax<double>(teq::Shape({1}), edge2, mvalues2);

		auto before2 = r2->data();
		eigen::RTMemptrT mem = memory;
		r2->assign(1, mem);
		double* raw2 = (double*) r2->data();
		ASSERT_NE(nullptr, raw2);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before2);
		EXPECT_EQ(outdata2.data(), raw2);
#endif

		EXPECT_EQ(4, raw2[0]);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, SparseArgMax)
{
	auto memory = std::make_shared<MockRuntimeMemory>();
	size_t lifetimes = 0;
	auto incr_life = [&lifetimes]{ ++lifetimes; };

	teq::RanksT ranks = {0, 1};
	std::vector<double> orig_raw = {
		0, 2, 0,
		0, 4, 6,
	};
	teq::Shape outshape({3, 2});

	MockEigen mockdev;
	std::vector<eigen::StorageIdxT> inner_indices;
	std::vector<eigen::StorageIdxT> outer_indices;
	auto edge = make_sparse_var(orig_raw, inner_indices,
		outer_indices, mockdev, outshape, "", incr_life);
	for (teq::RankT rank : ranks)
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RankT>().pack(mvalues, rank);

		teq::DimT outdim;
		teq::Shape outshape;
		if (rank == 0)
		{
			outdim = 2;
			outshape = {1, 2};
		}
		else
		{
			outdim = 3;
			outshape = {3};
		}
		std::vector<double> outdata(6);

		{
#ifndef PERM_OP
			auto outbytes = outdim * sizeof(double);
			EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
			EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
			auto r = eigen::argmax<double>(outshape, edge, mvalues);

			auto before = r->data();
			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			double* raw = (double*) r->data();
			ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
			EXPECT_EQ(nullptr, before);
			EXPECT_EQ(outdata.data(), raw);
#endif

			if (0 == rank)
			{
				EXPECT_EQ(1, raw[0]);
				EXPECT_EQ(2, raw[1]);
			}
			else
			{
				EXPECT_EQ(0, raw[0]);
				EXPECT_EQ(1, raw[1]);
				EXPECT_EQ(1, raw[2]);
			}
			EXPECT_EQ(1, lifetimes);
		}
		lifetimes = 0;
	}

	marsh::Maps mvalues2;
	eigen::Packer<teq::RankT>().pack(mvalues2, 8);

	std::vector<double> orig_raw2 = {
		0, 2, 0,
		0, 4, 9,
	};
	MockEigen mockdev2;
	auto edge2 = make_sparse_var(orig_raw2, inner_indices, outer_indices,
		mockdev2, teq::Shape({3, 2}), "", incr_life);

	std::vector<double> outdata2(6);
	{
#ifndef PERM_OP
		auto outbytes2 = sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes2)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes2)).Times(1);
#endif
		auto r2 = eigen::argmax<double>(teq::Shape({1}), edge2, mvalues2);

		auto before2 = r2->data();
		eigen::RTMemptrT mem = memory;
		r2->assign(1, mem);
		double* raw2 = (double*) r2->data();
		ASSERT_NE(nullptr, raw2);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before2);
		EXPECT_EQ(outdata2.data(), raw2);
#endif

		EXPECT_EQ(5, raw2[0]);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Extend)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {1, 4});

	teq::Shape outshape({3, 4, 2});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	MockDeviceRef mockdev;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({3, 1, 2}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	std::vector<double> outdata(24);
	auto memory = std::make_shared<MockRuntimeMemory>();

	{
#ifndef PERM_OP
		auto outbytes = 24 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes));
#endif
		auto r = eigen::extend<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 8, 4, 2, 8, 4, 2, 8, 4, 2, 8, 4,
			5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, MatrixExtend)
{
	std::vector<double> orig_raw{2, 8, 4, 5};
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockDeviceRef mockdev;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({2, 2}));

	std::vector<double> expect_raw = {
		2, 8, 2, 8,
		4, 5, 4, 5,
		2, 8, 2, 8,
		4, 5, 4, 5,
	};
	std::vector<double> expect_raw0 = {
		2, 8, 2, 8,
		4, 5, 4, 5,
	};
	std::vector<double> expect_raw1 = {
		2, 8,
		4, 5,
		2, 8,
		4, 5,
	};
	teq::Shape outshape({4, 4});
	teq::Shape outshape0({4, 2});
	teq::Shape outshape1({2, 4});

	std::vector<size_t> scenarios = {0, 1, 2};
	for (size_t scenario : scenarios)
	{
		marsh::Maps mvalues;
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		teq::Shape oshape;
		std::vector<double> expect;
		switch (scenario)
		{
			case 0:
				expect = expect_raw;
				oshape = outshape;
				eigen::Packer<teq::DimsT>().pack(mvalues, {2, 2});
				break;
			case 1:
				expect = expect_raw0;
				oshape = outshape0;
				eigen::Packer<teq::DimsT>().pack(mvalues, {2});
				break;
			case 2:
				expect = expect_raw1;
				oshape = outshape1;
				eigen::Packer<teq::DimsT>().pack(mvalues, {1, 2});
				break;
		}

		size_t nout = expect.size();
		std::vector<double> outdata(nout);

		{
#ifndef PERM_OP
			auto outbytes = nout * sizeof(double);
			EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
			EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes));
#endif
			auto r = eigen::extend<double>(oshape, edge, mvalues);

			auto before = r->data();
			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			double* raw = (double*) r->data();
			ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
			EXPECT_EQ(nullptr, before);
			EXPECT_EQ(outdata.data(), raw);
#endif

			std::vector<double> got_raw(raw, raw + oshape.n_elems());
			EXPECT_VECEQ(expect, got_raw);
			EXPECT_EQ(1, lifetimes);
		}
	}
}


TEST(OPERATOR, SparseExtend)
{
	std::vector<double> orig_raw{
		2, 0,
		0, 4,
	};
	std::vector<eigen::StorageIdxT> inner;
	std::vector<eigen::StorageIdxT> outer;
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockEigen mockdev;
	auto edge = make_sparse_var(orig_raw, inner, outer,
		mockdev, teq::Shape({2, 2}));

	std::vector<double> expect_raw = {
		2, 0, 2, 0,
		0, 4, 0, 4,
		2, 0, 2, 0,
		0, 4, 0, 4,
	};
	std::vector<double> expect_raw0 = {
		2, 0, 2, 0,
		0, 4, 0, 4,
	};
	std::vector<double> expect_raw1 = {
		2, 0,
		0, 4,
		2, 0,
		0, 4,
	};
	teq::Shape outshape({4, 4});
	teq::Shape outshape0({4, 2});
	teq::Shape outshape1({2, 4});

	std::vector<size_t> scenarios = {0, 1, 2};
	for (size_t scenario : scenarios)
	{
		marsh::Maps mvalues;
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		teq::Shape oshape;
		std::vector<double> expect;
		switch (scenario)
		{
			case 0:
				expect = expect_raw;
				oshape = outshape;
				eigen::Packer<teq::DimsT>().pack(mvalues, {2, 2});
				break;
			case 1:
				expect = expect_raw0;
				oshape = outshape0;
				eigen::Packer<teq::DimsT>().pack(mvalues, {2});
				break;
			case 2:
				expect = expect_raw1;
				oshape = outshape1;
				eigen::Packer<teq::DimsT>().pack(mvalues, {1, 2});
				break;
		}

		size_t nout = expect.size();
		std::vector<double> outdata(nout);

		{
			auto r = eigen::extend<double>(oshape, edge, mvalues);

			eigen::RTMemptrT mem = memory;
			r->assign(1, mem);
			expect_sparse_eq(expect, oshape, r);
		}
	}
}


TEST(OPERATOR, Permute)
{
	std::vector<double> outdata(12);
	std::vector<double> outdata2(12);
	std::vector<double> outdata3(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{2, 0, 1, 3, 4, 5, 6, 7});

		teq::Shape outshape({3, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12};
		MockDeviceRef mockdev;
		auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({2, 2, 3}));

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 12 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes));
#endif
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
	{ // same thing as above block except exclude 6 and 7 values
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{2, 0, 1, 3, 4, 5});

		teq::Shape outshape({3, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12};
		MockDeviceRef mockdev;
		auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({2, 2, 3}));

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 12 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes));
#endif
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata2.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{1, 0, 2, 3, 4, 5, 6, 7});

		teq::Shape outshape({3, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
		MockDeviceRef mockdev;
		auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({2, 3}));

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata3.data()));
		EXPECT_CALL(*memory, deallocate(outdata3.data(), outbytes));
#endif
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata3.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 4, 6, 8, 5, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Transpose)
{
	std::vector<double> outdata(12);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues,
			{1, 0, 2, 3, 4, 5, 6, 7});

		teq::Shape outshape({3, 4});
		std::vector<double> orig_raw{
			2, 8, 4, 5,
			6, 7, 1, 0,
			9, 11, 10, 12,
		};
		MockDeviceRef mockdev;
		auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({4, 3}));

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 12 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes));
#endif
		auto r = eigen::permute<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 6, 9,
			8, 7, 11,
			4, 1, 10,
			5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, SparseTranspose)
{
	auto memory = std::make_shared<MockRuntimeMemory>();
	marsh::Maps mvalues;
	eigen::Packer<teq::RanksT>().pack(mvalues,
		{1, 0, 2, 3, 4, 5, 6, 7});

	MockEigen mockdev;
	teq::Shape outshape({3, 2});
	std::vector<double> orig_raw{
		0, 8,
		5, 0,
		0, 7,
	};
	std::vector<eigen::StorageIdxT> inner;
	std::vector<eigen::StorageIdxT> outer;
	auto edge = make_sparse_var(orig_raw, inner, outer,
		mockdev, teq::Shape({2, 3}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	auto r = eigen::permute<double>(outshape, edge, mvalues);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);

	std::vector<double> expect_raw = {
		0, 5, 0,
		8, 0, 7,
	};
	expect_sparse_eq(expect_raw, outshape, r);
}


TEST(OPERATOR, Slice)
{
	std::vector<double> outdata(2);
	auto memory = std::make_shared<MockRuntimeMemory>();

	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	MockDeviceRef mockdev;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({3, 2}));
	// slice both dimensions 0 and 1
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{1, 2}, {1, 1}});

		teq::Shape outshape({2, 1});

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 2 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes));
#endif
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
	// slice last dimension (validate optimization)
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{0, 3}, {1, 1}});

		teq::Shape outshape({3, 1});
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);

		std::vector<double> expect_raw = {5, 6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, SparseSlice)
{
	std::vector<double> outdata(2);
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockEigen mockdev;
	std::vector<double> orig_raw{
		2, 0, 4,
		5, 0, 7,
	};
	std::vector<eigen::StorageIdxT> inner;
	std::vector<eigen::StorageIdxT> outer;
	auto edge = make_sparse_var(orig_raw, inner, outer,
		mockdev, teq::Shape({3, 2}));
	// slice both dimensions 0 and 1
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{1, 2}, {1, 1}});

		teq::Shape outshape({2, 1});
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);

		std::vector<double> expect_raw = {0, 7};
		expect_sparse_eq(expect_raw, outshape, r);
	}
	// disable optimization for sparse
	{
		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
			{{0, 3}, {1, 1}});

		teq::Shape outshape({3, 1});
		auto r = eigen::slice<double>(outshape, edge, mvalues);

		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);

		std::vector<double> expect_raw = {5, 0, 7};
		expect_sparse_eq(expect_raw, outshape, r);
	}
}


TEST(OPERATOR, Concat)
{
	std::vector<double> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	teq::Shape outshape({2, 4});
	std::vector<double> orig_raw{2, 8, 4, 5};
	std::vector<double> orig_raw2{1, 0, 3, 9};
	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({1, 4}));
	auto edge2 = make_var(orig_raw2.data(), mockdev2, teq::Shape({1, 4}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

#ifndef PERM_OP
	auto outbytes = 8 * sizeof(double);
	EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
	EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edge, edge2}, mvalues);

	auto before = r->data();
	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
	EXPECT_EQ(nullptr, before);
	EXPECT_EQ(outdata.data(), raw);
#endif

	std::vector<double> expect_raw = {
		2, 1,
		8, 0,
		4, 3,
		5, 9,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, ConcatMisMatchDim)
{
	std::vector<double> outdata(9);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);

	teq::Shape outshape({3, 3});
	std::vector<double> orig_rawa{2, 8, 4, 5, 7, 6};
	std::vector<double> orig_rawb{1, 0, 3};
	MockDeviceRef mockdeva;
	MockDeviceRef mockdevb;
	auto edgea = make_var(orig_rawa.data(), mockdeva, teq::Shape({2, 3}));
	auto edgeb = make_var(orig_rawb.data(), mockdevb, teq::Shape({1, 3}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
	[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
	[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

#ifndef PERM_OP
	auto outbytes = 9 * sizeof(double);
	EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
	EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edgea, edgeb}, mvalues);

	auto before = r->data();
	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
	EXPECT_EQ(nullptr, before);
	EXPECT_EQ(outdata.data(), raw);
#endif

	std::vector<double> expect_raw = {
		2, 8, 1,
		4, 5, 0,
		7, 6, 3,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, MultiConcat)
{
	std::vector<double> outdata(12);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	teq::Shape outshape({3, 4});
	std::vector<double> orig_raw{2, 8, 4, 5};
	std::vector<double> orig_raw2{1, 0, 3, 9};
	std::vector<double> orig_raw3{3, 7, 2, 11};
	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	MockDeviceRef mockdev3;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({1, 4}));
	auto edge2 = make_var(orig_raw2.data(), mockdev2, teq::Shape({1, 4}));
	auto edge3 = make_var(orig_raw3.data(), mockdev3, teq::Shape({1, 4}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev3), odata()).WillOnce(Invoke(
	[&orig_raw3, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw3.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

#ifndef PERM_OP
	auto outbytes = 12 * sizeof(double);
	EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
	EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edge, edge2, edge3}, mvalues);

	auto before = r->data();
	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
	EXPECT_EQ(nullptr, before);
	EXPECT_EQ(outdata.data(), raw);
#endif

	std::vector<double> expect_raw = {
		2, 1, 3,
		8, 0, 7,
		4, 3, 2,
		5, 9, 11,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
	EXPECT_EQ(3, lifetimes);
}


TEST(OPERATOR, BothSparseConcat)
{
	std::vector<double> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	teq::Shape outshape({2, 4});
	MockEigen mockdev;
	MockEigen mockdev2;
	std::vector<double> orig_raw{2, 0, 0, 5};
	std::vector<double> orig_raw2{0, 0, 3, 9};
	std::vector<eigen::StorageIdxT> inner1;
	std::vector<eigen::StorageIdxT> outer1;
	std::vector<eigen::StorageIdxT> inner2;
	std::vector<eigen::StorageIdxT> outer2;
	auto edge = make_sparse_var(orig_raw, inner1, outer1, mockdev, teq::Shape({1, 4}));
	auto edge2 = make_sparse_var(orig_raw2, inner2, outer2, mockdev2, teq::Shape({1, 4}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edge, edge2}, mvalues);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);

	std::vector<double> expect_raw = {
		2, 0,
		0, 0,
		0, 3,
		5, 9,
	};
	expect_sparse_eq(expect_raw, outshape, r);
}


TEST(OPERATOR, LeadingSparseConcat)
{
	std::vector<double> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	teq::Shape outshape({2, 4});
	MockEigen mockdev;
	MockDeviceRef mockdev2;
	std::vector<double> orig_raw{2, 0, 0, 5};
	std::vector<double> orig_raw2{1, 0, 3, 9};
	std::vector<eigen::StorageIdxT> inner1;
	std::vector<eigen::StorageIdxT> outer1;
	auto edge = make_sparse_var(orig_raw, inner1, outer1, mockdev, teq::Shape({1, 4}));
	auto edge2 = make_var(orig_raw2.data(), mockdev2, teq::Shape({1, 4}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

#ifndef PERM_OP
	auto outbytes = 8 * sizeof(double);
	EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
	EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edge, edge2}, mvalues);

	auto before = r->data();
	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
	EXPECT_EQ(nullptr, before);
	EXPECT_EQ(outdata.data(), raw);
#endif

	std::vector<double> expect_raw = {
		2, 1,
		0, 0,
		0, 3,
		5, 9,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, TrailingSparseConcat)
{
	std::vector<double> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RankT>().pack(mvalues, 0);
	teq::Shape outshape({2, 4});
	MockDeviceRef mockdev;
	MockEigen mockdev2;
	std::vector<double> orig_raw{1, 0, 3, 9};
	std::vector<double> orig_raw2{2, 0, 0, 5};
	std::vector<eigen::StorageIdxT> inner2;
	std::vector<eigen::StorageIdxT> outer2;
	auto edge = make_var(orig_raw.data(), mockdev, teq::Shape({1, 4}));
	auto edge2 = make_sparse_var(orig_raw2, inner2, outer2, mockdev2, teq::Shape({1, 4}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

#ifndef PERM_OP
	auto outbytes = 8 * sizeof(double);
	EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
	EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
	auto r = eigen::concat<double>(outshape, teq::TensptrsT{edge, edge2}, mvalues);

	auto before = r->data();
	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
	EXPECT_EQ(nullptr, before);
	EXPECT_EQ(outdata.data(), raw);
#endif

	std::vector<double> expect_raw = {
		1, 2,
		0, 0,
		3, 0,
		9, 5,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_VECEQ(expect_raw, got_raw);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, Pow)
{
	std::vector<double> outdata(6);
	std::vector<double> outdata2(8);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_raw2{1, 0, 3, 3, 2, 4};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::pow<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 1, 64, 125, 36, 2401};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 4, 2};
		std::vector<double> orig_raw2{1, 0, 3, 3, 2, 4, 2, 3};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes)).Times(1);
#endif
		auto r = eigen::pow<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata2.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 1, 64, 125, 36, 2401, 16, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Add)
{
	std::vector<double> outdata(8);
	std::vector<double> outdata2(6);
	std::vector<double> outdata3(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 8, 11};
		std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11, 6, 1.2};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::add<double>(outshape, teq::TensptrsT{edge, edge2});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18, 14, 12.2};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11};
	std::vector<double> orig_raw3{4.2, 1, 7.1, 1, 2, 1.1};
	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	MockDeviceRef mockdev3;
	auto edge = make_var(orig_raw.data(), mockdev, outshape);
	auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);
	auto edge3 = make_var(orig_raw3.data(), mockdev3, outshape);

	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes)).Times(1);
#endif
		auto r = eigen::add<double>(outshape, teq::TensptrsT{edge, edge2});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata2.data(), raw);
#endif

		std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev3), odata()).WillOnce(Invoke(
		[&orig_raw3, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw3.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata3.data()));
		EXPECT_CALL(*memory, deallocate(outdata3.data(), outbytes)).Times(1);
#endif
		auto r = eigen::add<double>(outshape, teq::TensptrsT{edge, edge2, edge3});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata3.data(), raw);
#endif

		std::vector<double> expect_raw = {7.2, 9, 14.1, 15, 18, 19.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(3, lifetimes);
	}
}


TEST(OPERATOR, Sub)
{
	std::vector<double> outdata(6);
	std::vector<double> outdata2(8);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::sub<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 8, 11};
		std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11, 6, 1.2};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata2.data()));
		EXPECT_CALL(*memory, deallocate(outdata2.data(), outbytes)).Times(1);
#endif
		auto r = eigen::sub<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata2.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4, 2, 9.8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Mul)
{
	std::vector<double> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 1.2, 3};
		std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11, 2, 1.7};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77, 2.4, 5.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}

	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	std::vector<double> orig_raw2{1, 0, 3, 9, 10, 11};
	std::vector<double> orig_raw3{4, 1, 7, 1, 2, 1};
	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	MockDeviceRef mockdev3;
	auto edge = make_var(orig_raw.data(), mockdev, outshape);
	auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);
	auto edge3 = make_var(orig_raw3.data(), mockdev3, outshape);
	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(6);
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev3), odata()).WillOnce(Invoke(
		[&orig_raw3, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw3.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(6);
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2, edge3});

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {8, 0, 84, 45, 120, 77};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(3, lifetimes);
	}
}


TEST(OPERATOR, BothSparseMul)
{
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockEigen mockdev;
	MockEigen mockdev2;
	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{
		2, 0,
		0, 5,
		6, 0,
	};
	std::vector<double> orig_raw2{
		1, 0,
		3, 0,
		10, 0,
	};
	std::vector<eigen::StorageIdxT> inner1;
	std::vector<eigen::StorageIdxT> outer1;
	std::vector<eigen::StorageIdxT> inner2;
	std::vector<eigen::StorageIdxT> outer2;
	auto edge = make_sparse_var(orig_raw, inner1, outer1, mockdev, outshape);
	auto edge2 = make_sparse_var(orig_raw2, inner2, outer2, mockdev2, outshape);
	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	std::vector<double> outdata(6);
	auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2});

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);

	std::vector<double> expect_raw = {
		2, 0,
		0, 0,
		60, 0,
	};
	expect_sparse_eq(expect_raw, outshape, r);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, LeadingSparseMul)
{
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockEigen mockdev;
	MockDeviceRef mockdev2;
	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{
		2, 0,
		0, 5,
		6, 0,
	};
	std::vector<double> orig_raw2{
		1, 0,
		3, 9,
		10, 11,
	};
	std::vector<eigen::StorageIdxT> inner1;
	std::vector<eigen::StorageIdxT> outer1;
	auto edge = make_sparse_var(orig_raw, inner1, outer1, mockdev, outshape);
	auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);
	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	std::vector<double> outdata(6);
	auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2});

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);

	std::vector<double> expect_raw = {
		2, 0,
		0, 45,
		60, 0,
	};
	expect_sparse_eq(expect_raw, outshape, r);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, TrailingSparseMul)
{
	auto memory = std::make_shared<MockRuntimeMemory>();

	MockDeviceRef mockdev;
	MockEigen mockdev2;
	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{
		1, 0,
		3, 9,
		10, 11,
	};
	std::vector<double> orig_raw2{
		2, 0,
		0, 5,
		6, 0,
	};
	std::vector<eigen::StorageIdxT> inner2;
	std::vector<eigen::StorageIdxT> outer2;
	auto edge = make_var(orig_raw.data(), mockdev, outshape);
	auto edge2 = make_sparse_var(orig_raw2, inner2, outer2, mockdev2, outshape);
	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	std::vector<double> outdata(6);
	auto r = eigen::mul<double>(outshape, teq::TensptrsT{edge, edge2});

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);

	std::vector<double> expect_raw = {
		2, 0,
		0, 45,
		60, 0,
	};
	expect_sparse_eq(expect_raw, outshape, r);
	EXPECT_EQ(2, lifetimes);
}


TEST(OPERATOR, Div)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_raw2{1, 0.5, 3, 9, 10, 11};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::div<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 16, 4./3, 5./9, 0.6, 7./11};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 6, 7, 1.2, 3};
		std::vector<double> orig_raw2{1, 0.5, 3, 9, 10, 11, 2, 1.7};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		auto edge = make_var(orig_raw.data(), mockdev, outshape);
		auto edge2 = make_var(orig_raw2.data(), mockdev2, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
		[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::div<double>(outshape, *edge, *edge2);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 16, 4./3, 5./9, 0.6, 7./11, 0.6, 3/1.7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Eq)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::eq<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {0, 0, 1, 0, 1, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::eq<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {0, 0, 1, 0, 1, 0, 1, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Neq)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::neq<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 1, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::neq<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 1, 0, 1, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Lt)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::lt<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {0, 0, 0, 1, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::lt<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {0, 0, 0, 1, 0, 1, 0, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Gt)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::gt<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 1, 0, 0, 0, 0};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::gt<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 1, 0, 0, 0, 0, 0, 1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Min)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::min<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 0.5, 4, 5, 6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::min<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 0.5, 4, 5, 6, 7, 3, 3};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
}


TEST(OPERATOR, Max)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::max<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 8, 4, 9, 6, 11};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_rawa{2, 8, 4, 5, 6, 7, 3, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::max<double>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {2, 8, 4, 9, 6, 11, 3, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);
	}
}


template <typename T>
void rand_uniform_test (
	const std::array<T,8>& adata,
	const std::array<T,8>& bdata)
{
	std::vector<T> outdata(8);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 4});
		std::vector<T> araw(adata.begin(), adata.end());
		std::vector<T> braw(bdata.begin(), bdata.end());
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(araw.data(), mockdeva, outshape);
		auto edgeb = make_var(braw.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&araw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(araw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&braw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(braw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::rand_uniform<T>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		T* raw = (T*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<T> got_raw(raw, raw + outshape.n_elems());
		for (size_t i = 0, n = got_raw.size(); i < n; ++i)
		{
			T e = got_raw[i];
			EXPECT_LE(adata[i], e);
			EXPECT_GE(bdata[i], e);
		}
		EXPECT_EQ(2, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<T> araw(adata.begin(), adata.end());
		std::vector<T> braw(bdata.begin(), bdata.end());
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(araw.data(), mockdeva, outshape);
		auto edgeb = make_var(braw.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&araw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(araw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&braw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(braw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<T> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::rand_uniform<T>(outshape, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		T* raw = (T*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<T> got_raw(raw, raw + outshape.n_elems());
		for (size_t i = 0, n = got_raw.size(); i < n; ++i)
		{
			T e = got_raw[i];
			EXPECT_LE(adata[i], e);
			EXPECT_GE(bdata[i], e);
		}
		EXPECT_EQ(2, lifetimes);
	}
}


TEST(OPERATOR, RandUniformInt)
{
	rand_uniform_test<size_t>({
		1, 2, 3, 5, 6, 7, 3, 8
	}, {
		2, 5, 4, 9, 9, 11, 4, 13
	});
}


TEST(OPERATOR, RandUniformDouble)
{
	rand_uniform_test<double>({
		1, 0.5, 3.5, 5, 6, 7, 3, 8
	}, {
		2, 0.75, 4, 9, 6.7, 11, 4, 13
	});
}


TEST(OPERATOR, Select)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		std::vector<double> orig_raw{0, 1, 0, 0, 1, 1};
		std::vector<double> orig_rawa{2, 8, 9, 5, 8, 7};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11};
		MockDeviceRef mockdevcomp;
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto comp = make_var(orig_raw.data(), mockdevcomp, outshape);
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdevcomp), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::select<double>(outshape, *comp, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 8, 4, 9, 8, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(3, lifetimes);
	}
	{
		teq::Shape outshape({2, 2, 2});
		std::vector<double> orig_raw{0, 1, 0, 0, 1, 1, 0, 1};
		std::vector<double> orig_rawa{2, 8, 9, 5, 8, 7, 4, 8};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 3};
		MockDeviceRef mockdevcomp;
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto comp = make_var(orig_raw.data(), mockdevcomp, outshape);
		auto edgea = make_var(orig_rawa.data(), mockdeva, outshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, outshape);
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdevcomp), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(8);
#ifndef PERM_OP
		auto outbytes = 8 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::select<double>(outshape, *comp, *edgea, *edgeb);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {1, 8, 4, 9, 8, 7, 3, 8};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(3, lifetimes);
	}
}


TEST(OPERATOR, Contract)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		teq::Shape outshape({2, 3});
		teq::Shape lshape({4, 3});
		teq::Shape rshape({2, 4});
		std::vector<double> orig_rawa{2, 8, 9, 5, 8, 7, 1, 9, 4.2, 3, 2, 6};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 8};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(orig_rawa.data(), mockdeva, lshape);
		auto edgeb = make_var(orig_rawb.data(), mockdevb, rshape);

		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().pack(mvalues, {{0, 1}});

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::contract<double>(outshape, *edgea, *edgeb, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {103, 212, 69, 150, 46.2, 99.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({2, 3});
		teq::Shape lshape({4, 2, 3});
		teq::Shape rshape({2, 4, 2});
		std::vector<double> orig_rawa{2, 8, 9, 5, 8, 7, 1, 9, 4.2, 3, 2, 6, 2, 8, 9, 5, 8, 7, 1, 9, 4.2, 3, 2, 6};
		std::vector<double> orig_rawb{1, 0.5, 4, 9, 6, 11, 3, 8, 1, 0.5, 4, 9, 6, 11, 3, 8};
		MockDeviceRef mockdeva;
		MockDeviceRef mockdevb;
		auto edgea = make_var(lshape);
		auto edgeb = make_var(rshape);
		EXPECT_CALL(*edgea, device()).WillRepeatedly(ReturnRef(mockdeva));
		EXPECT_CALL(*edgeb, device()).WillRepeatedly(ReturnRef(mockdevb));
		EXPECT_CALL(Const(*edgea), device()).WillRepeatedly(ReturnRef(mockdeva));
		EXPECT_CALL(Const(*edgeb), device()).WillRepeatedly(ReturnRef(mockdevb));

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
		[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));
		EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
		[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		marsh::Maps mvalues;
		eigen::Packer<eigen::PairVecT<teq::RankT>>().pack(mvalues,
			{{0, 1}, {1, 2}});

		std::vector<double> outdata(6);
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::contract<double>(outshape, *edgea, *edgeb, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {172, 362, 149.2, 311.1, 115.2, 249.1};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes); // called once per argument
	}
}


TEST(OPERATOR, SparseMatmul)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();
	MockEigen mockdeva;
	MockEigen mockdevb;
	teq::Shape outshape({2, 3});
	teq::Shape lshape({4, 3});
	teq::Shape rshape({2, 4});
	std::vector<double> orig_rawa{
		2, 0, 0, 5,
		0, 7, 0, 0,
		0, 0, 0, 3,
	};
	std::vector<double> orig_rawb{
		1, 0,
		0, 9,
		0, 0,
		0, 8,
	};
	std::vector<eigen::StorageIdxT> inner_indicesa;
	std::vector<eigen::StorageIdxT> outer_indicesa;
	std::vector<eigen::StorageIdxT> inner_indicesb;
	std::vector<eigen::StorageIdxT> outer_indicesb;
	auto edgea = make_sparse_var(orig_rawa,
		inner_indicesa, outer_indicesa, mockdeva, lshape);
	auto edgeb = make_sparse_var(orig_rawb,
		inner_indicesb, outer_indicesb, mockdevb, rshape);

	marsh::Maps mvalues;
	eigen::Packer<eigen::PairVecT<teq::RankT>>().pack(mvalues, {{0, 1}});

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdeva), odata()).WillOnce(Invoke(
	[&orig_rawa, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_rawa.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdevb), odata()).WillOnce(Invoke(
	[&orig_rawb, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_rawb.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	auto r = eigen::contract<double>(outshape, *edgea, *edgeb, mvalues);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);

	std::vector<double> expect_raw = {
		2, 40,
		0, 63,
		0, 24,
	};
	expect_sparse_eq(expect_raw, outshape, r);
}


TEST(OPERATOR, Pad)
{
	marsh::Maps mvalues;
	eigen::Packer<eigen::PairVecT<teq::DimT>>().pack(mvalues,
		{{1, 1}});

	teq::Shape outshape({4, 3});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	MockLeaf edge;
	MockDeviceRef mockdev;
	make_var(edge, orig_raw.data(), mockdev, teq::Shape({2, 3}));

	std::vector<double> outdata(12);
	auto memory = std::make_shared<MockRuntimeMemory>();

	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 12 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::pad<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			0, 2, 8, 0,
			0, 4, 5, 0,
			0, 6, 7, 0,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Stride)
{
	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {1, 2});

	teq::Shape outshape({2, 2});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	MockLeaf edge;
	MockDeviceRef mockdev;
	make_var(edge, orig_raw.data(), mockdev, teq::Shape({2, 3}));

	std::vector<double> outdata(4);
	auto memory = std::make_shared<MockRuntimeMemory>();

	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&orig_raw, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 4 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::stride<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 8, 6, 7,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Scatter)
{
	std::vector<double> outdata(9);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::DimsT>().pack(mvalues, {2, 2});

	teq::Shape outshape({3, 3});
	std::vector<double> orig_raw{2, 8, 4, 5};
	MockLeaf edge;
	MockDeviceRef mockdev;
	make_var(edge, orig_raw.data(), mockdev, teq::Shape({2, 2}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	{
#ifndef PERM_OP
		auto outbytes = 9 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::scatter<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2, 0, 8,
			0, 0, 0,
			4, 0, 5,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Reverse)
{
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<std::set<teq::RankT>>().pack(mvalues, {1});

	teq::Shape outshape({2, 3});
	std::vector<double> orig_raw{2, 8, 4, 5, 6, 7};
	MockDeviceRef mockdev;
	auto edge = make_var(orig_raw.data(), mockdev, outshape);

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	{
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::reverse<double>(outshape, edge, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			6, 7, 4, 5, 2, 8
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_VECEQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
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
	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();

	std::vector<double> expect;
	std::transform(invec.begin(), invec.end(), std::back_inserter(expect),
	[&](double e){ return unary(e); });
	{
		teq::Shape shape({2, 3});
		MockLeaf edge;
		MockDeviceRef mockdev;
		make_var(edge, invec.data(), mockdev, shape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&invec, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(invec.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = f(shape, edge);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
	{
		teq::Shape shape({2, 1, 3});
		MockLeaf edge;
		MockDeviceRef mockdev;
		make_var(edge, invec.data(), mockdev, shape);

		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&invec, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(invec.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

		std::vector<double> outdata(6);
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = f(shape, edge);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


TEST(OPERATOR, Abs)
{
	elementary_unary(eigen::abs<double>, [](double e){ return std::abs(e); });
}


TEST(OPERATOR, Neg)
{
	elementary_unary(eigen::neg<double>, [](double e){ return -e; });
}


TEST(OPERATOR, Sin)
{
	elementary_unary(eigen::sin<double>, [](double e){ return std::sin(e); });
}


TEST(OPERATOR, Cos)
{
	elementary_unary(eigen::cos<double>, [](double e){ return std::cos(e); });
}


TEST(OPERATOR, Tan)
{
	elementary_unary(eigen::tan<double>, [](double e){ return std::tan(e); });
}


TEST(OPERATOR, Exp)
{
	elementary_unary(eigen::exp<double>, [](double e){ return std::exp(e); });
}


TEST(OPERATOR, Log)
{
	elementary_unary(eigen::log<double>, [](double e){ return std::log(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Sqrt)
{
	elementary_unary(eigen::sqrt<double>, [](double e){ return std::sqrt(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Round)
{
	elementary_unary(eigen::round<double>, [](double e){ return std::round(e); },
		{3.22, 8.51, 2.499, 5.2, 7.17, 3.79});
}


TEST(OPERATOR, Sigmoid)
{
	elementary_unary(
	[](teq::Shape outshape, const teq::iTensor& in)
	{ return eigen::sigmoid<double>(outshape, in); },
	[](double e){ return 1. / (1. + std::exp(-e)); });
}


TEST(OPERATOR, Tanh)
{
	elementary_unary(
	[](teq::Shape outshape, const teq::iTensor& in)
	{ return eigen::tanh<double>(outshape, in); },
	[](double e){ return std::tanh(e); });
}


TEST(OPERATOR, Square)
{
	elementary_unary(eigen::square<double>,
	[](double e){ return e * e; });
}


TEST(OPERATOR, Cube)
{
	elementary_unary(
	[](teq::Shape outshape, const teq::iTensor& in)
	{ return eigen::cube<double>(outshape, in); },
	[](double e){ return e * e * e; });
}


TEST(OPERATOR, Convolution)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(An<const std::string&>())).WillRepeatedly(Return(false));

	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues, {1, 1});
		teq::Shape outshape({3, 2});
		std::vector<double> orig_raw{
			2, 8, 4,
			5, 7, 6,
			9, 1, 0,
		};
		std::vector<double> orig_raw2{0.3, 0.6};
		MockLeaf image;
		MockLeaf kernel;
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		make_var(image, orig_raw.data(), mockdev, teq::Shape({3, 3}));
		make_var(kernel, orig_raw2.data(), mockdev2, teq::Shape({2}));

		std::string fatalmsg = "convolution does not support repeated kernel dimensions: [1\\1]";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));;
		EXPECT_FATAL(eigen::convolution<double>(outshape, image, kernel, mvalues), fatalmsg.c_str());
	}

	{
		marsh::Maps mvalues;
		eigen::Packer<teq::RanksT>().pack(mvalues, {1});
		teq::Shape outshape({3, 2});
		std::vector<double> orig_raw{2, 8, 4, 5, 7, 6, 9, 1, 0};
		std::vector<double> orig_raw2{0.3, 0.6, 4.0, 2.2};
		MockDeviceRef mockdev;
		MockDeviceRef mockdev2;
		MockLeaf image;
		MockLeaf kernel;
		make_var(image, orig_raw.data(), mockdev, teq::Shape({3, 3}));
		make_var(kernel, orig_raw2.data(), mockdev2, teq::Shape({2, 2}));

		std::string fatalmsg1 = "given kernel shape [2\\2], unspecified non-singular kernel dimension 1 is undefined";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));;
		EXPECT_FATAL(eigen::convolution<double>(outshape, image, kernel, mvalues), fatalmsg1.c_str());
	}

	std::vector<double> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();

	marsh::Maps mvalues;
	eigen::Packer<teq::RanksT>().pack(mvalues, {1});

	teq::Shape outshape({3, 2});
	std::vector<double> orig_raw{2, 8, 4, 5, 7, 6, 9, 1, 0};
	std::vector<double> orig_raw2{0.3, 0.6};
	MockDeviceRef mockdev;
	MockDeviceRef mockdev2;
	MockLeaf image;
	MockLeaf kernel;
	make_var(image, orig_raw.data(), mockdev, teq::Shape({3, 3}));
	make_var(kernel, orig_raw2.data(), mockdev2, teq::Shape({2}));

	size_t lifetimes = 0;
	EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
	[&orig_raw, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));
	EXPECT_CALL(Const(mockdev2), odata()).WillOnce(Invoke(
	[&orig_raw2, &lifetimes]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(orig_raw2.data(), [&lifetimes]{ ++lifetimes; });
		return out;
	}));

	{
#ifndef PERM_OP
		auto outbytes = 6 * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::convolution<double>(outshape, image, kernel, mvalues);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<double> expect_raw = {
			2 * 0.3 + 5 * 0.6, 8 * 0.3 + 7 * 0.6, 4 * 0.3 + 6 * 0.6,
			5 * 0.3 + 9 * 0.6, 7 * 0.3 + 1 * 0.6, 6 * 0.3 + 0 * 0.6,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		ASSERT_ARRDBLEQ(expect_raw, got_raw);
		EXPECT_EQ(2, lifetimes);

		global::set_logger(new exam::NoSupportLogger());
	}
}


TEST(OPERATOR, Assign)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	MockMutableLeaf edgea;
	MockLeaf edgeb;
	eigen::SrcRef<double> devref(a.data(), outshape);
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	EXPECT_CALL(edgea, shape()).WillRepeatedly(Return(outshape));
	EXPECT_CALL(edgea, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(edgea), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(edgea, upversion(1)).Times(1);
	make_var(edgeb, b.data(), mockdev, outshape);
	EXPECT_CALL(edgeb, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(0));

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::assign<double>(edgea, edgeb);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
	EXPECT_EQ(raw, devref.data());

	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(b, got_raw);
}


TEST(OPERATOR, AssignAdd)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	MockMutableLeaf edgea;
	MockLeaf edgeb;
	eigen::SrcRef<double> devref(a.data(), outshape);
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	EXPECT_CALL(edgea, shape()).WillRepeatedly(Return(outshape));
	EXPECT_CALL(edgea, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(edgea), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(edgea, upversion(1)).Times(1);
	make_var(edgeb, b.data(), mockdev, outshape);
	EXPECT_CALL(edgeb, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(0));

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::assign_add<double>(edgea, edgeb);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
	EXPECT_EQ(raw, devref.data());

	std::vector<double> expect_raw = {3, 8, 7, 14, 16, 18};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignSub)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	MockMutableLeaf edgea;
	MockLeaf edgeb;
	eigen::SrcRef<double> devref(a.data(), outshape);
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	EXPECT_CALL(edgea, shape()).WillRepeatedly(Return(outshape));
	EXPECT_CALL(edgea, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(edgea), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(edgea, upversion(1)).Times(1);
	make_var(edgeb, b.data(), mockdev, outshape);
	EXPECT_CALL(edgeb, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(0));

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::assign_sub<double>(edgea, edgeb);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
	EXPECT_EQ(raw, devref.data());

	std::vector<double> expect_raw = {1, 8, 1, -4, -4, -4};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignMul)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 0, 3, 9, 10, 11};
	MockMutableLeaf edgea;
	MockLeaf edgeb;
	eigen::SrcRef<double> devref(a.data(), outshape);
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	EXPECT_CALL(edgea, shape()).WillRepeatedly(Return(outshape));
	EXPECT_CALL(edgea, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(edgea), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(edgea, upversion(1)).Times(1);
	make_var(edgeb, b.data(), mockdev, outshape);
	EXPECT_CALL(edgeb, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(0));

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::assign_mul<double>(edgea, edgeb);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
	EXPECT_EQ(raw, devref.data());

	std::vector<double> expect_raw = {2, 0, 12, 45, 60, 77};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, AssignDiv)
{
	teq::Shape outshape({2, 3});
	std::vector<double> a{2, 8, 4, 5, 6, 7};
	std::vector<double> b{1, 2, 3, 9, 10, 11};
	MockMutableLeaf edgea;
	MockLeaf edgeb;
	eigen::SrcRef<double> devref(a.data(), outshape);
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	EXPECT_CALL(edgea, shape()).WillRepeatedly(Return(outshape));
	EXPECT_CALL(edgea, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(edgea), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(edgea, upversion(1)).Times(1);
	make_var(edgeb, b.data(), mockdev, outshape);
	EXPECT_CALL(edgeb, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillRepeatedly(Return(0));

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);

	auto r = eigen::assign_div<double>(edgea, edgeb);

	eigen::RTMemptrT mem = memory;
	r->assign(1, mem);
	double* raw = (double*) r->data();
	ASSERT_NE(nullptr, raw);
	EXPECT_EQ(raw, devref.data());

	std::vector<double> expect_raw = {2, 4, 4./3, 5./9, 0.6, 7./11};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


TEST(OPERATOR, Cast)
{
	std::vector<int32_t> outdata(6);
	auto memory = std::make_shared<MockRuntimeMemory>();

	teq::Shape outshape({2, 3});
	std::vector<double> a{2.1, 8.5, 4.3, 5.2, 6.1, 7.2};
	MockDeviceRef mockdev;
	MockMeta mockmeta;
	auto edgea = make_var(a.data(), mockdev, outshape);
	EXPECT_CALL(*edgea, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return(egen::name_type(egen::DOUBLE)));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));

	{
		EXPECT_CALL(*memory, allocate(_)).Times(0);

		auto r = eigen::cast<double>(edgea);

		double* raw = (double*) r->data();
		ASSERT_NE(nullptr, raw); // assert data existing at the beginning due to ptr ref
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);

		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		ASSERT_ARRDBLEQ(a, got_raw);
	}
	{
		size_t lifetimes = 0;
		EXPECT_CALL(Const(mockdev), odata()).WillOnce(Invoke(
		[&a, &lifetimes]() -> teq::Once<const void*>
		{
			teq::Once<const void*> out(a.data(), [&lifetimes]{ ++lifetimes; });
			return out;
		}));

#ifndef PERM_OP
		auto outbytes = 6 * sizeof(int32_t);
		EXPECT_CALL(*memory, allocate(outbytes)).WillOnce(Return(outdata.data()));
		EXPECT_CALL(*memory, deallocate(outdata.data(), outbytes)).Times(1);
#endif
		auto r = eigen::cast<int32_t>(edgea);

		auto before = r->data();
		eigen::RTMemptrT mem = memory;
		r->assign(1, mem);
		int32_t* raw = (int32_t*) r->data();
		ASSERT_NE(nullptr, raw);
#ifndef PERM_OP
		EXPECT_EQ(nullptr, before);
		EXPECT_EQ(outdata.data(), raw);
#endif

		std::vector<int32_t> expect_raw = {2, 8, 4, 5, 6, 7};
		std::vector<int32_t> got_raw(raw, raw + outshape.n_elems());
		ASSERT_ARREQ(expect_raw, got_raw);
		EXPECT_EQ(1, lifetimes);
	}
}


#endif // DISABLE_EIGEN_OPERATOR_TEST
