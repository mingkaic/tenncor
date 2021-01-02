
#ifndef DISABLE_EIGEN_DEVICE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mock.hpp"


using ::testing::Return;
using ::testing::ReturnRef;


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
	std::vector<double> destdata = {2, 3, 7, 2};
	eigen::SrcRef<double> devref(destdata.data(), shape);
	auto dest = std::make_shared<MockMutableLeaf>();
	EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*dest, device()).WillRepeatedly(ReturnRef(devref));

	auto srcvar = make_var(shape);

	bool assign_called = false;
	eigen::TensAssign<double> ref(*dest, *srcvar,
	[&devref, &assign_called, &srcvar](eigen::TensorT<double>& dst, const teq::iTensor& src)
	{
		EXPECT_EQ(devref.data(), dst.data());
		EXPECT_EQ(srcvar.get(), &src);
		assign_called = true;
	});
	ref.assign(); // assigning shouldn't do anything
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


TEST(DEVICE, Calc)
{
	teq::Shape shape({3});
	std::vector<double> data = {1, 2, 3};
	std::vector<double> srcdata = {2, 3, 7, 2};
	std::vector<double> srcdata2 = {2, 8, 4};
	std::vector<double> srcdata3 = {3, 7, 5};
	auto dest = std::make_shared<MockMutableLeaf>();
	eigen::PtrRef<double> srcref(srcdata.data());
	eigen::PtrRef<double> srcref2(srcdata2.data());
	eigen::PtrRef<double> srcref3(srcdata3.data());
	EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*dest, device()).WillRepeatedly(ReturnRef(srcref));

	eigen::Device dev;

	auto lhs = make_var(shape);
	auto rhs = make_var(shape);
	EXPECT_CALL(*lhs, device()).WillRepeatedly(ReturnRef(srcref2));
	EXPECT_CALL(*rhs, device()).WillRepeatedly(ReturnRef(srcref3));

	eigen::PtrRef<double> src(data.data());
	auto obsref = std::make_shared<MockEigen>();
	MockMeta mockmeta;	

	auto obs = std::make_shared<MockObservable>();
	EXPECT_CALL(*obs, get_args()).WillRepeatedly(Return(teq::TensptrsT{lhs, rhs}));
	EXPECT_CALL(*obs, device()).WillRepeatedly(ReturnRef(*obsref));
	EXPECT_CALL(*obs, to_string()).WillRepeatedly(Return("Hello"));

	EXPECT_CALL(*obsref, assign()).Times(1);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(true));
	dev.calc(*obs);

	dev.max_version_ = 0;
	EXPECT_CALL(*obsref, assign()).Times(0);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(false));
	dev.calc(*obs);

	dev.max_version_ = 10;
	EXPECT_CALL(*obsref, assign()).Times(1);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(true));
	dev.calc(*obs);
}


#endif // DISABLE_EIGEN_DEVICE_TEST
