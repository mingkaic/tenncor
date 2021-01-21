#ifndef EIGEN_MOCK_MEMORY_HPP
#define EIGEN_MOCK_MEMORY_HPP

#include "internal/eigen/eigen.hpp"

#include "gmock/gmock.h"

struct MockRuntimeMemory final : public eigen::iRuntimeMemory
{
	MOCK_METHOD1(allocate, void*(size_t));
	MOCK_METHOD2(deallocate, void(void*,size_t));
};

#endif // EIGEN_MOCK_MEMORY_HPP
