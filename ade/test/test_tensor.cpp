
#ifndef DISABLE_TENSOR_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "common.hpp"

#include "ade/cmap.hpp"


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
	ade::TensptrT tens(new MockTensor(ade::Shape(slist)));
	ade::MappedTensor mt = ade::flip_map(tens, dim);

	ade::Shape shape = mt.shape();
	EXPECT_ARREQ(slist, shape);

	ade::MappedTensor mt2(tens, ade::CoordptrT(new ade::CoordMap(
		[](ade::MatrixT m)
		{
			for (size_t i = 0; i < ade::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[0][0] = 4;
		})));

	ade::Shape shape2 = mt2.shape();
	EXPECT_EQ(4 * slist[0], shape2.at(0));

	EXPECT_FATAL(ade::identity_map(nullptr),
		"cannot map a null tensor");

	EXPECT_FATAL(ade::MappedTensor(nullptr, ade::reduce(3, {4}),
		false, ade::extend(3, {4})),
		"cannot map a null tensor");
}


#endif // DISABLE_TENSOR_TEST
