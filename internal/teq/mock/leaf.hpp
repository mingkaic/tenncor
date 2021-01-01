
#ifndef TEQ_MOCK_LEAF_HPP
#define TEQ_MOCK_LEAF_HPP

#include "internal/teq/teq.hpp"

#include "internal/teq/mock/device.hpp"

#include "gmock/gmock.h"

using ::testing::Const;
using ::testing::Return;
using ::testing::ReturnRef;

struct MockLeaf final : public teq::iLeaf
{
	MOCK_CONST_METHOD0(to_string, std::string(void));
	MOCK_METHOD0(device, teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(device, const teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(get_meta, const teq::iMetadata&(void));
	MOCK_CONST_METHOD0(shape, teq::Shape(void));
	MOCK_CONST_METHOD0(clone_impl, teq::iTensor*(void));

	MOCK_CONST_METHOD0(get_usage, teq::Usage(void));
};

using MockLeafptrT = std::shared_ptr<MockLeaf>;

void make_var (MockLeaf& out, const teq::Shape& shape, const std::string& label = "");

template <typename T>
void make_var (MockLeaf& out, T* data, MockDeviceRef& devref,
	const teq::Shape& shape, const std::string& label = "")
{
	make_var(out, shape, label);
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(data));
	EXPECT_CALL(Const(devref), data()).WillRepeatedly(Return(data));

	EXPECT_CALL(out, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(out), device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(out, get_usage()).WillRepeatedly(Return(teq::IMMUTABLE));
}

MockLeafptrT make_var (const teq::Shape& shape, const std::string& label = "");

template <typename T>
MockLeafptrT make_var (T* data, MockDeviceRef& devref,
	const teq::Shape& shape, const std::string& label = "")
{
	auto out = make_var(shape, label);
	make_var<T>(*out, data, devref, shape, label);
	return out;
}

template <typename T>
MockLeafptrT make_cst (T& data, MockDeviceRef& devref)
{
	return make_var<T>(&data, devref, teq::Shape(), fmts::to_string(data));
}

#endif // TEQ_MOCK_LEAF_HPP
