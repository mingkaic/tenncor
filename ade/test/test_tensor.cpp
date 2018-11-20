
#ifndef DISABLE_TENSOR_TEST


#include "gtest/gtest.h"

#include "ade/tensor.hpp"

#include "testutil/common.hpp"

#include "common.hpp"


struct TENSOR : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(TENSOR, Tensorptr)
{
	std::weak_ptr<ade::iTensor> weaks;
	{
		ade::Tensor* raw = new MockTensor();
		std::shared_ptr<ade::iTensor> shared(new MockTensor());
		ade::Tensorptr ptr(raw);
		ade::Tensorptr sharedtens(shared);
		{
			ade::Tensorptr optr(ptr);
			EXPECT_EQ(ptr.get(), optr.get());
			weaks = optr.ref();
		}
		EXPECT_FALSE(weaks.expired());
		EXPECT_EQ(raw, ptr.get());
		EXPECT_EQ(shared.get(), sharedtens.get());
	}
	EXPECT_TRUE(weaks.expired());

	EXPECT_FATAL(ade::Tensorptr(nullptr), "cannot create nodeptr with nullptr");
	EXPECT_FATAL(ade::Tensorptr(std::shared_ptr<ade::iTensor>(nullptr)),
		"cannot create nodeptr with nullptr");
}


TEST_F(TENSOR, MappedTensor)
{
	simple::SessionT sess = get_session("TENSOR::MappedTensor");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");

	size_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});
	ade::CoordPtrT flipper = ade::flip(dim);
	
	ade::Tensorptr tens(new MockTensor(ade::Shape(slist)));
	ade::MappedTensor mt(flipper, tens);

	ade::Shape shape = mt.shape();
	EXPECT_ARREQ(slist, shape);

	mt.mapper_ = ade::CoordPtrT(new ade::CoordMap(
		[](ade::MatrixT m)
		{
			for (size_t i = 0; i < ade::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[0][0] = 4;
		}));

	ade::Shape shape2 = mt.shape();
	EXPECT_EQ(4 * slist[0], shape2.at(0));
}


#endif // DISABLE_TENSOR_TEST
