//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef TENNCOR_MOCK_ITENSOR_H
#define TENNCOR_MOCK_ITENSOR_H

#include "tests/unit/include/utils/util_test.hpp"

#include "include/tensor/tensor_double.hpp"


// randomly initiates raw data on construction
class mock_itensor : public tensor_double
{
public:
	mock_itensor (void) : tensor_double() {}

	mock_itensor (FUZZ::fuzz_test* fuzzer, tensorshape shape, std::vector<double> initdata = {}) :
		tensor_double(shape)
	{
		if (is_alloc())
		{
			size_t n = get_alloced_shape().n_elems();
			if (initdata.empty())
			{
				initdata = fuzzer->get_double(n, "initdata", {-123, 139.2});
			}
			set_data(&initdata[0], n * sizeof(double));
		}
	}

	mock_itensor (double scalar) :
		tensor_double(scalar) {}

	mock_itensor* clone (bool shapeonly = false) const
	{
		return static_cast<mock_itensor*>(clone_impl(shapeonly));
	}

	mock_itensor* move (void)
	{
		return static_cast<mock_itensor*>(move_impl());
	}

	mock_itensor& operator = (const mock_itensor& other)
	{
		tensor_double::operator = (other);
		return *this;
	}

	mock_itensor& operator = (mock_itensor&& other)
	{
		tensor_double::operator = (std::move(other));
		return *this;
	}

	// checks if two tensors are equal without exposing
	bool equal (const mock_itensor& other) const
	{
		// check shape equality
		if (false == is_alloc() ||
			false == other.is_alloc())
		{
			return tensorshape_equal(get_shape(), other.get_shape());
		}
		if (false == get_alloced_shape().is_compatible_with(other.get_alloced_shape()))
		{
			return false;
		}

		// check
		size_t n = this->n_elems();
		// crashes if we have shape, data inconsistency,
		// assuming address sanitation works properly
		const double* data = (const double*) get_data();
		const double* other_data = (const double*) other.get_data();
		return std::equal(data, data + n, other_data);
	}

	// checks if get_alloced_shape() is undefined when not allocated
	bool clean (void) const
	{
		// checks by ensuring data is null and alloc is undefined when unallocated
		return is_alloc() || (
			nullptr == get_data() &&
			false == get_alloced_shape().is_part_defined());
	}

	const double* rawptr (void) const { return (const double*) get_data(); }

	bool allocshape_is (const tensorshape& shape)
	{
		return tensorshape_equal(get_alloced_shape(), shape);
	}

protected:
	mock_itensor (const mock_itensor& other, bool shapeonly = false) :
		tensor_double(other, shapeonly) {}

	mock_itensor (mock_itensor&& other) :
		tensor_double(std::move(other)) {}

	virtual tensor* clone_impl (bool shapeonly) const
	{
		return new mock_itensor(*this, shapeonly);
	}

	virtual tensor* move_impl (void)
	{
		return new mock_itensor(std::move(*this));
	}
};


#endif //TENNCOR_MOCK_ITENSOR_H

