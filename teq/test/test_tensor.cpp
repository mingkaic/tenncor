
#ifndef DISABLE_TENSOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"

#include "teq/funcarg.hpp"


TEST(TENSOR, FuncArg)
{
	std::vector<teq::DimT> slist = {2, 81};

	size_t dim = 1;
	teq::TensptrT tens(new MockTensor(teq::Shape(slist)));
	teq::FuncArg mt = teq::flip_map(tens, dim);

	teq::Shape shape = mt.shape();
	EXPECT_ARREQ(slist, shape);

	teq::FuncArg mt2(tens, teq::CoordptrT(new teq::CoordMap(
		[](teq::MatrixT& m)
		{
			for (size_t i = 0; i < teq::mat_dim; ++i)
			{
				m[i][i] = 1;
			}
			m[0][0] = 4;
		})));

	teq::Shape shape2 = mt2.shape();
	EXPECT_EQ(4 * slist[0], shape2.at(0));

	EXPECT_FATAL(teq::identity_map(nullptr),
		"cannot map a null tensor");

	EXPECT_FATAL(teq::FuncArg(nullptr, teq::reduce(3, {4}),
		false, teq::extend(3, {4})),
		"cannot map a null tensor");
}


#endif // DISABLE_TENSOR_TEST
