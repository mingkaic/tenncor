//
//  itensor.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-15.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/itensor.hpp"

#ifdef TENNCOR_ITENSOR_HPP

namespace nnet
{

itensor* itensor::clone (bool shapeonly) const
{
	return clone_impl(shapeonly);
}

itensor* itensor::move (void)
{
	return move_impl();
}

size_t itensor::rank (void) const
{
	return get_shape().rank();
}

std::vector<size_t> itensor::dims (void) const
{
	return get_shape().as_list();
}

bool itensor::is_same_size (const itensor& other) const
{
	if (is_alloc() && other.is_alloc())
	{
		tensorshape simp_shape = get_alloced_shape().trim();
		tensorshape other_simp = other.get_alloced_shape().trim();
		return simp_shape.is_compatible_with(other_simp);
	}

	return get_allowed_shape().is_compatible_with(other.get_allowed_shape());
}

bool itensor::is_compatible_with (const itensor& other) const
{
	return get_shape().is_compatible_with(other.get_shape());
}

bool itensor::is_compatible_with (size_t ndata) const
{
	const tensorshape& my_shape = get_shape();

	bool compatible = true;
	// perfect fit
	if (my_shape.is_fully_defined())
	{
		compatible = ndata == my_shape.n_elems();
	}
	else
	{
		size_t known = my_shape.n_known();
		if (0 < known)
		{
			compatible = 0 == ndata % known;
		}
	}

	return compatible;
}

bool itensor::is_loosely_compatible_with (size_t ndata) const
{
	const tensorshape& my_shape = get_shape();

	bool compatible = true;
	if (my_shape.is_fully_defined())
	{
		compatible = ndata <= my_shape.n_elems();
	}
	// partially defined shapes are always compatible,
	// since unknown dimension can expand infinitely to fit data
	return compatible;
}

optional<tensorshape> itensor::guess_shape (size_t ndata) const
{
	optional<tensorshape> bestshape;
	const tensorshape& allowed_shape = get_allowed_shape();
	// if allowed is fully defined
	if (allowed_shape.is_fully_defined())
	{
		if (allowed_shape.n_elems() == ndata)
		{
			bestshape = allowed_shape;
		}
		return bestshape;
	}
	// if allowed is partially defined
	else if (allowed_shape.is_part_defined())
	{
		std::vector<size_t> my_shape = allowed_shape.as_list();
		size_t rank = my_shape.size();
		size_t first_undef = my_shape.size();
		size_t known = 1;
		for (size_t i = 0; i < rank; i++)
		{
			if (0 == my_shape[i])
			{
				if (first_undef> i)
				{
					first_undef = i;
				}
				my_shape[i] = 1;
			}
			else
			{
				known *= my_shape[i];
			}
		}
		assert(known> 0);
		if (0 == ndata % known)
		{
			my_shape[first_undef] = ndata / known;
			bestshape = tensorshape(my_shape);
		}
	}
	// if allowed is undefined
	else
	{
		bestshape = tensorshape({ndata});
	}
	return bestshape;
}

optional<tensorshape> itensor::loosely_guess_shape (size_t ndata) const
{
	const tensorshape& allowed_shape = get_allowed_shape();
	if (allowed_shape.is_fully_defined())
	{
		optional<tensorshape> bestshape;
		if (allowed_shape.n_elems()>= ndata)
		{
			bestshape = allowed_shape;
		}
		return bestshape;
	}
	std::vector<size_t> my_shape = allowed_shape.as_list();
	size_t first_undef = my_shape.size();
	size_t known = 1;
	for (size_t i = 0; i < my_shape.size(); i++)
	{
		if (0 == my_shape[i])
		{
			if (first_undef> i)
			{
				first_undef = i;
			}
			my_shape[i] = 1;
		}
		else
		{
			known *= my_shape[i];
		}
	}
	my_shape[first_undef] = ndata / known;
	if (0 != ndata % known)
	{
		// int division above will floor
		// (if we cast to double, we may lose precision)
		my_shape[first_undef]++;
	}
	return tensorshape(my_shape);
}

bool itensor::is_aligned (void) const
{
	return true;
}

void itensor::serialize (tenncor::tensor_proto* proto) const
{
	if (false == is_alloc()) return;
	// copy bytes
	size_t nb = total_bytes();
	proto->set_data(get_data(), nb);

	proto->set_type(get_type());

	std::vector<size_t> allowed = get_allowed_shape().as_list();
	std::vector<size_t> alloced = get_alloced_shape().as_list();
	google::protobuf::RepeatedField<uint64_t> allowed_field(allowed.begin(), allowed.end());
	google::protobuf::RepeatedField<uint64_t> alloced_field(alloced.begin(), alloced.end());

	proto->mutable_allowed_shape()->Swap(&allowed_field);
	proto->mutable_alloced_shape()->Swap(&alloced_field);
}

bool itensor::from_proto (const tenncor::tensor_proto& other)
{
	if (get_type() != other.type())
	{
		throw std::exception(); // incompatible types
	}

	// shapes must have same dimensionality... (otherwise, input data is definitely corrupt)
	assert(other.alloced_shape_size() == other.allowed_shape_size());
	std::vector<size_t> allowed(other.allowed_shape().begin(), other.allowed_shape().end());
	std::vector<size_t> alloced(other.alloced_shape().begin(), other.alloced_shape().end());
	tensorshape allowed_shape = tensorshape(allowed);
	set_allowed_shape(allowed_shape);
	tensorshape temp_alloc_shape(alloced);
	// another sanity check, be less stringent, since this may represent some less evident issue
	if (false == temp_alloc_shape.is_compatible_with(allowed_shape) ||
		false == temp_alloc_shape.is_fully_defined()) return false;

	deallocate();
	set_alloced_shape(temp_alloc_shape);
	assert(allocate());

	// copy data over from tensor_proto
	std::string protostr = other.data();
	set_data((void*) protostr.c_str(), protostr.size());

	return true;
}

bool itensor::from_proto (const tenncor::tensor_proto& other, size_t alloc_id)
{
	set_allocator(alloc_id);
	return from_proto(other);
}

size_t type_size (tenncor::tensor_proto::tensor_t type)
{
	switch (type)
	{
		case tenncor::tensor_proto::DOUBLE_T:
			return sizeof(double);
		case tenncor::tensor_proto::SIGNED_T:
			return sizeof(signed);
		default:
		break;
	}
	return 0;
}

template <>
tenncor::tensor_proto::tensor_t get_prototype<double> (void)
{
	return tenncor::tensor_proto::DOUBLE_T;
}

template <>
tenncor::tensor_proto::tensor_t get_prototype<signed> (void)
{
	return tenncor::tensor_proto::SIGNED_T;
}

}

#endif
