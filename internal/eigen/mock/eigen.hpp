#ifndef EIGEN_MOCK_EIGEN_HPP
#define EIGEN_MOCK_EIGEN_HPP

#include "internal/eigen/eigen.hpp"

#include "gmock/gmock.h"

struct MockEigen final : public eigen::iEigen
{
	MOCK_METHOD0(data, void*(void));
	MOCK_CONST_METHOD0(data, const void*(void));

	MOCK_METHOD0(assign, void(void));
};

#endif // EIGEN_MOCK_EIGEN_HPP
