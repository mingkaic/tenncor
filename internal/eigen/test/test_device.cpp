
#ifndef DISABLE_EIGEN_DEVICE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mock.hpp"


TEST(DEVICE, SrcRef)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};
	eigen::SrcRef<double> ref(data.data(), shape);

	[&data](const eigen::iEigen& ref)
	{
		auto ptr = (double*) ref.data();
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(data, vec);
	}(ref);

	ref.assign(); // assigning shouldn't do anything

	auto ptr = (double*) ref.data();
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(data, vec);
}


TEST(DEVICE, PtrRef)
{
	std::vector<double> data = {1, 2, 3, 4};
	eigen::PtrRef<double> ref(data.data());
}


TEST(DEVICE, TensAssign)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};
	auto dest = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, shape);

	eigen::TensAssign<double,std::vector<double>> ref(*dest, data,
	[](eigen::TensorT<double>& dst, std::vector<double>& src)
	{
		//
	});
}


TEST(DEVICE, TensAccum)
{}


TEST(DEVICE, TensOp)
{}


TEST(DEVICE, MatOp)
{}


#endif // DISABLE_EIGEN_DEVICE_TEST
