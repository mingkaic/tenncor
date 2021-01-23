
#ifndef TEQ_MOCK_DEVICEREF_HPP
#define TEQ_MOCK_DEVICEREF_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

using ::testing::Const;
using ::testing::Invoke;
using ::testing::Return;

struct MockDeviceRef final : public teq::iDeviceRef
{
	MOCK_METHOD0(data, void*(void));
	MOCK_CONST_METHOD0(data, const void*(void));
	MOCK_METHOD0(odata, teq::Once<void*>(void));
	MOCK_CONST_METHOD0(odata, teq::Once<const void*>(void));
};

struct MockDevice final : public teq::iDevice
{
	MOCK_METHOD2(calc, void(teq::iTensor&,size_t));
};

template <typename T>
void make_devref (MockDeviceRef& devref, T* data,
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

#endif // TEQ_MOCK_DEVICEREF_HPP
