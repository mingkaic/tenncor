#ifndef EIGEN_MOCK_EIGEN_HPP
#define EIGEN_MOCK_EIGEN_HPP

#include "internal/eigen/eigen.hpp"

#include "gmock/gmock.h"

struct MockEigen final : public eigen::iEigen
{
	MOCK_METHOD0(data, void*(void));
	MOCK_CONST_METHOD0(data, const void*(void));
	MOCK_METHOD0(odata, teq::Once<void*>(void));
	MOCK_CONST_METHOD0(odata, teq::Once<const void*>(void));

	MOCK_CONST_METHOD0(sparse_info, eigen::OptSparseT(void));
	MOCK_METHOD2(assign, void(size_t,eigen::RTMemptrT&));
	MOCK_CONST_METHOD1(valid_for, bool(size_t));
	MOCK_METHOD1(extend_life, void(size_t));
	MOCK_METHOD0(expire, void());
};

template <typename T>
void make_eigen (MockEigen& devref, T* data,
	jobs::GuardOpF once_exec = jobs::GuardOpF())
{
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(data));
	EXPECT_CALL(Const(devref), data()).WillRepeatedly(Return(data));
	EXPECT_CALL(devref, odata()).WillRepeatedly(Invoke(
	[data, once_exec]() -> teq::Once<void*>
	{
		teq::Once<void*> out(data, once_exec);
		return out;
	}));
	EXPECT_CALL(Const(devref), odata()).WillRepeatedly(Invoke(
	[data, once_exec]() -> teq::Once<const void*>
	{
		teq::Once<const void*> out(data, once_exec);
		return out;
	}));
}

template <typename T>
void make_var (MockLeaf& out, T* data, MockEigen& devref,
	const teq::Shape& shape, const std::string& label = "",
	jobs::GuardOpF once_exec = jobs::GuardOpF())
{
	make_var(out, shape, label);
	make_eigen(devref, data, once_exec);

	EXPECT_CALL(out, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(out), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(out, get_usage()).WillRepeatedly(Return(teq::IMMUTABLE));
}

template <typename T>
MockLeafptrT make_var (T* data, MockEigen& devref,
	const teq::Shape& shape, const std::string& label = "",
	jobs::GuardOpF once_exec = jobs::GuardOpF())
{
	auto out = std::make_shared<MockLeaf>();
	make_var<T>(*out, data, devref, shape, label, once_exec);
	return out;
}

#endif // EIGEN_MOCK_EIGEN_HPP
