/*!
 *
 *  placeholder.hpp
 *  cnnet
 *
 *  Purpose:
 *  placeholder implementation
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <list>
#include <new>
#include <memory>

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_PLACEHOLDER_HPP
#define TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

class placeholder final : public ileaf
{
public:
	//! shape constructor
	placeholder (const tensorshape& shape, std::string name = "");

	//! explicitly declare copy constructor since assignments are declared
	placeholder (const placeholder& other);

	//! explicitly declare move constructor since assignments are declared
	placeholder (placeholder&& other);

	//! clone function
	placeholder* clone (void) const;

	//! move function
	placeholder* move (void);

	//! declare copy assignment to avoid implicit deletion
	virtual placeholder& operator = (const placeholder& other);

	//! declare move assignment to avoid implicit deletion
	virtual placeholder& operator = (placeholder&& other);



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

	// >>>>>> PLACEHOLDEr SPECIAL <<<<<<

	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	template <typename T>
	placeholder& operator = (std::vector<T> data)
	{
		std::shared_ptr<void> ptr = &data[0];
		TENS_TYPE type = data_->get_type();
		if (false == data_->has_data())
		{
			asgn_.set_data(ptr, type, data_->get_shape(), 0);
			data_->read_from(asgn_);
		}
		else
		{
			if (optional<tensorshape> shape = data_->guess_shape(data.size()))
			{
				asgn_.set_data(ptr, type, *shape, 0);
				data_->read_from(asgn_, *shape);
			}
			// we would reach here if data is empty... (todo: test. currently never reached)
			else
			{
				throw std::logic_error("attempting to assign no data to an unallocated tensor");
			}
		}
		this->notify(UPDATE);
		return *this;
	}

	//! assign tensor to inner tensor
	virtual placeholder& operator = (tensor& data);

protected:
	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	void copy_helper (const placeholder& other);

	void move_helper (placeholder&& other);

private:
	assign_io asgn_;

	//! raw data
	std::unique_ptr<tensor> data_ = nullptr;
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
	placeptr& operator = (tensor& ten);

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
