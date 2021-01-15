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

	MOCK_METHOD2(assign, void(size_t,eigen::iRuntimeMemory&));
	MOCK_CONST_METHOD1(valid_for, bool(size_t));
	MOCK_METHOD1(extend_life, void(size_t));
};

#endif // EIGEN_MOCK_EIGEN_HPP
