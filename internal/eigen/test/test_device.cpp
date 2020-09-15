
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

	ref.assign(); // referencing shouldn't do anything

	auto ptr = (double*) ref.data();
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(data, vec);
}


TEST(DEVICE, PtrRef)
{
	std::vector<double> data = {1, 2, 3, 4};
	eigen::PtrRef<double> ref(data.data());

	[&data](const eigen::iEigen& ref)
	{
		auto ptr = (double*) ref.data();
		EXPECT_EQ(data.data(), ptr);
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(data, vec);
	}(ref);

	ref.assign(); // referencing shouldn't do anything

	auto ptr = (double*) ref.data();
	EXPECT_EQ(data.data(), ptr);
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(data, vec);
}


TEST(DEVICE, TensAssign)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};
	auto dest = std::make_shared<MockMutableLeaf>(
		std::vector<double>{2, 3, 7, 2}, shape);

	bool assign_called = false;
	eigen::TensAssign<double,std::vector<double>> ref(*dest, data,
	[&dest, &data, &assign_called](eigen::TensorT<double>& dst, std::vector<double>& src)
	{
		EXPECT_EQ(dest->device().data(), dst.data());
		EXPECT_VECEQ(data, src);
		assign_called = true;
	});

	[&dest](const eigen::iEigen& ref)
	{
		EXPECT_EQ(dest->device().data(), ref.data());
	}(ref);

	ref.assign(); // assigning shouldn't do anything

	EXPECT_EQ(dest->device().data(), ref.data());
	EXPECT_TRUE(assign_called);
}


TEST(DEVICE, TensAccum)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};

	bool assign_called = false;
	eigen::TensAccum<double,std::vector<double>> ref(
	4., eigen::shape_convert(shape), data,
	[&data, &assign_called](eigen::TensorT<double>& dst, const std::vector<double>& src)
	{
		double* ptr = dst.data();
		std::vector<double> expect(4, 4);
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(expect, vec);
		EXPECT_VECEQ(data, src);
		assign_called = true;
	});

	[](const eigen::iEigen& ref)
	{
		auto ptr = (double*) ref.data();
		std::vector<double> expect(4, 0);
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(expect, vec);
	}(ref);

	ref.assign(); // should initialize

	auto ptr = (double*) ref.data();
	std::vector<double> expect(4, 4);
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(expect, vec);
	EXPECT_TRUE(assign_called);
}


TEST(DEVICE, TensOp)
{
	teq::Shape shape({2, 1, 2});
	std::vector<double> data = {1, 2, 3, 4};

	bool init_called = false;
	eigen::TensOp<double,eigen::TensorT<double>,std::vector<double>>
	ref(eigen::shape_convert(shape), data,
	[&shape, &data, &init_called](std::vector<double>& src)
	{
		EXPECT_VECEQ(data, src);
		eigen::TensorT<double> out(eigen::shape_convert(shape));
		out.setConstant(5);
		init_called = true;
		return out;
	});
	EXPECT_TRUE(init_called);
	init_called = false;

	[](const eigen::iEigen& ref)
	{
		auto ptr = (double*) ref.data();
		std::vector<double> expect(4, 0);
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(expect, vec);
	}(ref);

	ref.assign(); // should initialize

	auto ptr = (double*) ref.data();
	std::vector<double> expect(4, 5);
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(expect, vec);
	EXPECT_FALSE(init_called);
}


TEST(DEVICE, MatOp)
{
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};

	bool init_called = false;
	eigen::MatOp<double,eigen::MatrixT<double>,std::vector<double>>
	ref(eigen::shape_convert(shape), data,
	[&shape, &data, &init_called](std::vector<double>& src)
	{
		EXPECT_VECEQ(data, src);
		eigen::MatrixT<double> out(2, 2);
		out.setConstant(6);
		init_called = true;
		return out;
	});
	EXPECT_TRUE(init_called);
	init_called = false;

	[](const eigen::iEigen& ref)
	{
		auto ptr = (double*) ref.data();
		std::vector<double> expect(4, 0);
		std::vector<double> vec(ptr, ptr + 4);
		EXPECT_VECEQ(expect, vec);
	}(ref);

	ref.assign(); // should initialize

	auto ptr = (double*) ref.data();
	std::vector<double> expect(4, 6);
	std::vector<double> vec(ptr, ptr + 4);
	EXPECT_VECEQ(expect, vec);
	EXPECT_FALSE(init_called);
}


#endif // DISABLE_EIGEN_DEVICE_TEST
