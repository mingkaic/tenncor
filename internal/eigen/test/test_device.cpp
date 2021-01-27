
#ifndef DISABLE_EIGEN_DEVICE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/eigen/mock/mock.hpp"


using ::testing::_;
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

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);
	eigen::RTMemptrT mem = memory;
	ref.assign(0, mem); // referencing shouldn't do anything

	auto ptr = (double*) ref.data();
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
	EXPECT_CALL(*dest, upversion(1)).Times(1);

	MockDeviceRef srcref;
	MockMeta mockmeta;
	auto srcvar = make_var(data.data(), srcref, shape);
	EXPECT_CALL(*srcvar, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, state_version()).WillOnce(Return(0));

	bool assign_called = false;
	eigen::TensAssign<double> ref(*dest, *srcvar,
	[&devref, &srcref, &assign_called](eigen::TensMapT<double>& dst, const eigen::TensMapT<double>& src)
	{
		EXPECT_EQ(devref.data(), dst.data());
		EXPECT_EQ(srcref.data(), src.data());
		assign_called = true;
	});

	auto memory = std::make_shared<MockRuntimeMemory>();
	EXPECT_CALL(*memory, allocate(_)).Times(0);
	eigen::RTMemptrT mem = memory;
	ref.assign(0, mem); // assigning shouldn't do anything
	EXPECT_TRUE(assign_called);
}


TEST(DEVICE, TensOp)
{
	teq::Shape outshape({3, 1, 2});
	teq::Shape shape({2, 1, 2});
	std::vector<double> data = {1, 2, 3, 4};
	std::vector<double> alloc_mem = {1, 2, 3, 4, 5, 6};

	size_t lifetimes = 0;
	auto incr_life = [&lifetimes]{ ++lifetimes; };

	MockDeviceRef devref;
	auto var = make_var(data.data(), devref, shape, "", incr_life);

	auto memory = std::make_shared<MockRuntimeMemory>();
	{
		bool init_called = false;
		eigen::TensOp<double> ref(outshape, teq::CTensT{var.get()},
		[&alloc_mem, &data, &init_called](
			eigen::TensMapT<double>& out,
			const std::vector<eigen::TensMapT<double>>& args)
		{
			ASSERT_EQ(1, args.size());
			EXPECT_EQ(alloc_mem.data(), out.data());
			EXPECT_EQ(data.data(), args[0].data());
			init_called = true;
		});
		EXPECT_FALSE(init_called);
		EXPECT_EQ(nullptr, ref.data());

		auto outbytes = outshape.n_elems() * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).Times(1).WillOnce(Return(alloc_mem.data()));
		EXPECT_CALL(*memory, deallocate(alloc_mem.data(), outbytes)).Times(1);
		eigen::RTMemptrT mem = memory;
		ref.assign(1, mem); // should initialize
		EXPECT_TRUE(init_called);
	}
}


TEST(DEVICE, MatOp)
{
	teq::Shape outshape({3, 2});
	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};
	std::vector<double> alloc_mem = {1, 2, 3, 4, 5, 6};

	size_t lifetimes = 0;
	auto incr_life = [&lifetimes]{ ++lifetimes; };

	MockDeviceRef devref;
	auto var = make_var(data.data(), devref, shape, "", incr_life);

	auto memory = std::make_shared<MockRuntimeMemory>();

	{
		bool init_called = false;
		eigen::MatOp<double> ref(outshape, teq::CTensT{var.get()},
		[&alloc_mem, &data, &init_called](
			eigen::MatMapT<double>& out,
			const std::vector<eigen::MatMapT<double>>& args)
		{
			ASSERT_EQ(1, args.size());
			EXPECT_EQ(alloc_mem.data(), out.data());
			EXPECT_EQ(data.data(), args[0].data());
			init_called = true;
		});
		EXPECT_FALSE(init_called);
		EXPECT_EQ(nullptr, ref.data());

		auto outbytes = outshape.n_elems() * sizeof(double);
		EXPECT_CALL(*memory, allocate(outbytes)).Times(1).WillOnce(Return(alloc_mem.data()));
		EXPECT_CALL(*memory, deallocate(alloc_mem.data(), outbytes)).Times(1);
		eigen::RTMemptrT mem = memory;
		ref.assign(1, mem); // should initialize
		EXPECT_TRUE(init_called);
	}
}


TEST(DEVICE, Calc)
{
	teq::Shape shape({3});
	std::vector<double> data = {1, 2, 3};
	std::vector<double> srcdata = {2, 3, 7, 2};
	std::vector<double> srcdata2 = {2, 8, 4};
	std::vector<double> srcdata3 = {3, 7, 5};
	auto dest = std::make_shared<MockMutableLeaf>();
	MockDeviceRef srcref, srcref2, srcref3;
	make_devref(srcref, srcdata.data());
	make_devref(srcref2, srcdata2.data());
	make_devref(srcref3, srcdata3.data());
	EXPECT_CALL(*dest, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*dest, device()).WillRepeatedly(ReturnRef(srcref));

	eigen::Device dev;

	auto lhs = make_var(shape);
	auto rhs = make_var(shape);
	EXPECT_CALL(*lhs, device()).WillRepeatedly(ReturnRef(srcref2));
	EXPECT_CALL(*rhs, device()).WillRepeatedly(ReturnRef(srcref3));

	MockDeviceRef src;
	make_devref(src, data.data());
	auto obsref = std::make_shared<MockEigen>();
	MockMeta mockmeta;
	EXPECT_CALL(*obsref, valid_for(0)).WillOnce(Return(false));
	EXPECT_CALL(*obsref, extend_life(0));

	double mockdata = 0;
	EXPECT_CALL(*obsref, data()).WillRepeatedly(Return(&mockdata));
	EXPECT_CALL(Const(*obsref), data()).WillRepeatedly(Return(&mockdata));

	auto obs = std::make_shared<MockObservable>();
	EXPECT_CALL(*obs, get_args()).WillRepeatedly(Return(teq::TensptrsT{lhs, rhs}));
	EXPECT_CALL(*obs, device()).WillRepeatedly(ReturnRef(*obsref));
	EXPECT_CALL(*obs, to_string()).WillRepeatedly(Return("Hello"));

	EXPECT_CALL(*obsref, assign(1, _)).Times(1);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(true));
	dev.calc(*obs,0);

	dev.max_version_ = 0;
	EXPECT_CALL(*obsref, assign(_, _)).Times(0);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(false));
	dev.calc(*obs,0);

	dev.max_version_ = 10;
	EXPECT_CALL(*obsref, assign(1, _)).Times(1);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(true));
	dev.calc(*obs,0);

	MockMObservable parent({obs});
	MockMObservable parent2({obs});
	EXPECT_CALL(*obsref, assign(2, _)).Times(1);
	EXPECT_CALL(*obs, prop_version(dev.max_version_)).Times(1).WillOnce(Return(true));
	dev.calc(*obs, 0);
}


#endif // DISABLE_EIGEN_DEVICE_TEST
