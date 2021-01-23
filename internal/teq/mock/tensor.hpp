
#ifndef TEQ_MOCK_TENSOR_HPP
#define TEQ_MOCK_TENSOR_HPP

#include "internal/teq/teq.hpp"

#include "gmock/gmock.h"

struct MockTensor final : public teq::iTensor
{
	MOCK_CONST_METHOD0(to_string, std::string(void));
	MOCK_METHOD0(device, teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(device, const teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(get_meta, const teq::iMetadata&(void));
	MOCK_CONST_METHOD0(shape, teq::Shape(void));
	MOCK_CONST_METHOD0(clone_impl, teq::iTensor*(void));

	MOCK_METHOD1(accept, void(teq::iTraveler&));
};

#endif // TEQ_MOCK_TENSOR_HPP
