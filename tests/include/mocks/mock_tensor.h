//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef TENNCOR_MOCK_TENSOR_H
#define TENNCOR_MOCK_TENSOR_H

#include "tests/include/utils/util_test.h"

#include "include/tensor/tensor.hpp"


// randomly initiates raw data on construction
class mock_tensor : public tensor<double>
{
public:
	mock_tensor (void) : tensor<double>() {}

	mock_tensor (FUZZ::fuzz_test* fuzzer, tensorshape shape, std::vector<double> initdata = {}) :
		tensor<double>(shape)
	{
		if (is_alloc())
		{
			size_t n = alloced_shape_.n_elems();
			if (initdata.empty())
			{
				initdata = fuzzer->get_double(n, "initdata", {-123, 139.2});
			}
			std::memcpy(raw_data_, &initdata[0], n * sizeof(double));
		}
	}

	mock_tensor (double scalar) :
		tensor<double>(scalar) {}

	mock_tensor (const mock_tensor& other, bool shapeonly = false) :
		tensor<double>(other, shapeonly) {}

	mock_tensor (mock_tensor&& other) :
		tensor<double>(std::move(other)) {}

	mock_tensor& operator = (const mock_tensor& other)
	{
		tensor<double>::operator = (other);
		return *this;
	}

	mock_tensor& operator = (mock_tensor&& other)
	{
		tensor<double>::operator = (std::move(other));
		return *this;
	}

	// checks if two tensors are equal without exposing
	bool equal (const mock_tensor& other) const
	{
		// check shape equality
		if (false == is_alloc() ||
			false == other.is_alloc())
		{
			return tensorshape_equal(get_shape(), other.get_shape());
		}
		if (false == alloced_shape_.is_compatible_with(other.alloced_shape_))
		{
			return false;
		}

		// check
		size_t n = alloced_shape_.n_elems();
		// crashes if we have shape, data inconsistency,
		// assuming address sanitation works properly
		return std::equal(raw_data_, raw_data_ + n, other.raw_data_);
	}

	// checks if alloced_shape_ is undefined when not allocated
	bool clean (void) const
	{
		// checks by ensuring data is null and alloc is undefined when unallocated
		return is_alloc() || (
			nullptr == raw_data_ &&
			false == alloced_shape_.is_part_defined());
	}

	const double* rawptr (void) const { return raw_data_; }

	bool allocshape_is (const tensorshape& shape)
	{
		return tensorshape_equal(alloced_shape_, shape);
	}
};


#endif //TENNCOR_MOCK_TENSOR_H

