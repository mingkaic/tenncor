/*!
 *
 *  itensor.hpp
 *  cnnet
 *
 *  Purpose:
 *  encapsulate tensor data type information
 *  and provide type-generic interface
 *
 *  Created by Mingkai Chen on 2017-03-10.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_ITENSOR_HPP
#define TENNCOR_ITENSOR_HPP

#include <vector>
#include <cstddef>

namespace nnet
{

class itensor
{
public:
	//! clone function
	itensor* clone (void) const { return clone_impl(); }

	// >>>> FUNDAMENTAL SHAPE INFO <<<<
	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	virtual size_t n_elems (void) const = 0;

	//! get the tensor rank, number of dimensions
	virtual size_t rank (void) const = 0;

	//! checks if tensorshape is aligned
	//! same number of column for each row
	virtual bool is_aligned (void) const = 0;

	// >>>> DATA INFORMATION <<<<
	//! checks if memory is allocated
	virtual bool is_alloc (void) const = 0;

	//! get bytes allocated
	virtual size_t total_bytes (void) const = 0;

	//! get data at coordinate specified
	virtual T get (std::vector<size_t> coord) const = 0;

	// >>>> DATA MUTATOR <<<<
	//! allocate raw data using innate shape
	virtual bool allocate (void) = 0;

	//! forcefully deallocate raw_data, invalidates external shape
	virtual bool deallocate (void) = 0;

	// >>>> SERIALIZATION HELPERS <<<<
	// void serialize (tenncor::tensor_proto* proto) const;
	// bool from_proto (const tensorproto& other);
	// bool from_proto (iallocator* a, const tensorproto& other);

protected:
	// >>>> ABSTRACT CLONE <<<<
	//! clone implementation
	virtual itensor* clone_impl (bool shapeonly) const = 0;
	
	//! move implementation
	virtual itensor* move_impl (void) = 0;
};

}

#endif /* TENNCOR_ITENSOR_HPP */
