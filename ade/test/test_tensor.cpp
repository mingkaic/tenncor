
#ifndef DISABLE_TENSOR_TEST


#include "gtest/gtest.h"

#include "ade/ileaf.hpp"

#include "testutil/common.hpp"

#include "common.hpp"


struct TENSOR : public ::testing::Test
{
	virtual void TearDown (void)
	{
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(TENSOR, MappedTensor)
{
	std::vector<ade::DimT> slist = {2, 81};

	size_t dim = 1;
	ade::CoordPtrT flipper = ade::flip(dim);

	ade::TensptrT tens(new MockTensor(ade::Shape(slist)));
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

	EXPECT_FATAL(ade::MappedTensor(ade::identity, nullptr),
		"cannot map a null tensor");
}


#endif // DISABLE_TENSOR_TEST
