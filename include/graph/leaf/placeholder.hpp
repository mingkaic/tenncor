/*!
 *
 *  placeholder.hpp
 *  cnnet
 *
 *  Purpose:
 *  placeholder implementation
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <list>
#include <new>
#include <memory>

#include "include/graph/leaf/ivariable.hpp"

#pragma once
#ifndef TENNCOR_PLACEHOLDER_HPP
#define TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

class placeholder final : public ivariable
{
public:
	// >>>> CONSTRUCTORS <<<<
	//! shape constructor
	placeholder (const tensorshape& shape, 
		tenncor::tensor_proto::tensor_t type = tenncor::tensor_proto::DOUBLE_T, 
		std::string name = "");

	//! explicitly declare copy constructor since assignments are declared
	placeholder (const placeholder& other);

	//! explicitly declare move constructor since assignments are declared
	placeholder (placeholder&& other);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	placeholder* clone (void) const;

	//! move function
	placeholder* move (void);

	//! declare copy assignment to avoid implicit deletion
	virtual placeholder& operator = (const placeholder& other);

	//! declare move assignment to avoid implicit deletion
	virtual placeholder& operator = (placeholder&& other);

	// >>>> DATA ASSIGNMENT OPERATORS <<<<
	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	template <typename T>
	placeholder& operator = (std::vector<T> data)
	{
		tenncor::tensor_proto::tensor_t type = this->data_->get_type();
		tenncor::tensor_proto::tensor_t ttype = get_prototype<T>();
		if (type != ttype)
		{
			throw std::exception(); // incompatible types
		}
		// note: if this is allocated,
		// compatibility is compared to allocated shape instead of allowed
		assert(this->data_->is_compatible_with(data.size()));

		if (false == this->data_->is_alloc())
		{
			if (optional<tensorshape> cand_shape = this->data_->guess_shape(data.size()))
			{
				this->data_->allocate(*cand_shape);
			}
			// we would reach here if data is empty... (todo: test. currently never reached)
			else
			{
				throw std::logic_error("attempting to assign no data to an unallocated tensor");
			}
		}
		this->assigner_(*(this->data_), &data[0], type);

		this->is_init_ = true;
		this->notify(UPDATE);
		return *this;
	}

	//! assign tensor to inner tensor
	virtual placeholder& operator = (itensor& data);

protected:
	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode* get_gradient (variable* );
};

class placeptr : public varptr
{
public:
	//! nullptr construction
	placeptr (void) {}

	//! wrap placeholder pointer
	placeptr (placeholder* ptr);

	//! assign a pointer
	placeptr& operator = (placeholder* other);

	// >>>> EXTENDING PLACEHOLDER <<<<
	//! assign a raw data
	template <typename T>
	placeptr& operator = (std::vector<T> vec)
	{
		get() = vec;
		return *this;
	}

	//! assign a tensor
	placeptr& operator = (itensor& ten);

	//! implicit pointer conversion
	operator placeholder* () const;

	//! dereference overload
	placeholder& operator * (void);

	//! pointer accessor overload
	placeholder* operator -> (void);

	//! get inner pointer as placeholder pointer
	placeholder* get (void) const;
};

}

#endif /* TENNCOR_PLACEHOLDER_HPP */
