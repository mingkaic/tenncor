
#ifndef EIGEN_MOCK_OBSERVABLE_HPP
#define EIGEN_MOCK_OBSERVABLE_HPP

#include "internal/eigen/eigen.hpp"

#include "internal/teq/mock/mock.hpp"

#include "gmock/gmock.h"

using ::testing::_;
using ::testing::Const;
using ::testing::Return;
using ::testing::ReturnRef;

struct MockObservable : public eigen::Observable
{
	MockObservable (void) : Observable(teq::TensptrsT{}) {}
	MockObservable (const teq::TensptrsT& args) : Observable(args) {}
	MockObservable (const teq::TensptrsT& args, marsh::Maps&& attrs) : Observable(args, std::move(attrs)) {}

	MOCK_CONST_METHOD0(to_string, std::string(void));
	MOCK_METHOD0(device, teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(device, const teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(get_meta, const teq::iMetadata&(void));
	MOCK_CONST_METHOD0(shape, teq::Shape(void));
	MOCK_CONST_METHOD0(clone_impl, teq::iTensor*(void));

	MOCK_CONST_METHOD0(get_opcode, teq::Opcode(void));
	MOCK_CONST_METHOD0(get_args, teq::TensptrsT(void));
	MOCK_METHOD2(update_child, void(teq::TensptrT,size_t));

	MOCK_CONST_METHOD0(has_data, bool(void));

	MOCK_METHOD0(uninitialize, void(void));

	MOCK_METHOD0(initialize, bool(void));

	MOCK_METHOD0(must_initialize, void(void));

	MOCK_METHOD1(prop_version, bool(size_t));
};

using MockObsptrT = std::shared_ptr<MockObservable>;

MockObsptrT make_obs (std::string opname, size_t opcode, teq::TensptrsT args);

MockObsptrT make_obs (std::string opname, size_t opcode, teq::TensptrsT args, marsh::Maps&& attrs);

template <typename T>
MockObsptrT make_obs (T* data, MockDeviceRef& devref,
	std::string opname, size_t opcode, teq::TensptrsT args)
{
	EXPECT_CALL(devref, data()).WillRepeatedly(Return(data));
	EXPECT_CALL(Const(devref), data()).WillRepeatedly(Return(data));

	auto out = make_obs(opname, opcode, args);
	EXPECT_CALL(*out, device()).WillRepeatedly(ReturnRef(devref));
	EXPECT_CALL(Const(*out), device()).WillRepeatedly(ReturnRef(devref));
	return out;
}

#endif // EIGEN_MOCK_OBSERVABLE_HPP
