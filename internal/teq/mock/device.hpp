
#ifndef TEQ_MOCK_DEVICEREF_HPP
#define TEQ_MOCK_DEVICEREF_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

struct MockDeviceRef final : public teq::iDeviceRef
{
	MOCK_METHOD0(data, void*(void));
	MOCK_CONST_METHOD0(data, const void*(void));
};

struct MockDevice final : public teq::iDevice
{
	MOCK_METHOD1(calc, void(teq::iTensor&));
};

#endif // TEQ_MOCK_DEVICEREF_HPP
