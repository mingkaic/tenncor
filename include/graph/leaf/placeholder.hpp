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

class placeholder final : public inode
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



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<inode*> get_leaves (void) const
	{
		return {const_cast<placeholder*>(this)};
	}



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
		size_t n = data.size();
		nnet::tensorshape inshape;
		if (data_->has_data())
		{
			if (data_->is_compatible_with(n))
			{
				inshape = data_->get_shape();
			}
			else
			{
				throw std::logic_error(nnutils::formatter() << "data with " 
					<< n << " elements cannot be assigned to allcoated tensor with " 
					<< data_->get_shape().n_elems() << " elements");
			}
		}
		else
		{
			if (optional<tensorshape> shape = data_->guess_shape(n))
			{
				inshape = *shape;
			}
			else
			{
				throw std::logic_error("attempting to assign badly shaped data to an unallocated tensor");
			}
		}
		size_t nbytes = n * sizeof(T);
		TENS_TYPE type = get_type<T>();
		std::shared_ptr<void> ptr = nnutils::make_svoid(nbytes);
		std::memcpy(ptr.get(), &data[0], nbytes);
		asgn_.set_data(ptr, type, inshape, 0);
		data_->read_from(asgn_, inshape);
	
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
