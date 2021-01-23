
#ifndef EIGEN_MOCK_MUTABLE_LEAF_HPP
#define EIGEN_MOCK_MUTABLE_LEAF_HPP

#include "internal/eigen/eigen.hpp"

#include "internal/teq/mock/mock.hpp"

#include "gmock/gmock.h"

struct MockMutableLeaf : public eigen::iMutableLeaf
{
	MOCK_CONST_METHOD0(to_string, std::string(void));
	MOCK_METHOD0(device, teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(device, const teq::iDeviceRef&(void));
	MOCK_CONST_METHOD0(get_meta, const teq::iMetadata&(void));
	MOCK_CONST_METHOD0(shape, teq::Shape(void));
	MOCK_CONST_METHOD0(clone_impl, teq::iTensor*(void));

	MOCK_CONST_METHOD0(get_usage, teq::Usage(void));

	MOCK_METHOD1(upversion, void(size_t));
};

#endif // EIGEN_MOCK_MUTABLE_LEAF_HPP
