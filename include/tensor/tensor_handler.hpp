/*!
 *
 *  tensor_handler.hpp
 *  cnnet
 *
 *  Purpose:
 *  handler is a delegate for manipulating raw datas in tensors
 *
 *  Created by Mingkai Chen on 2017-02-05.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include <typeinfo>

#include "include/tensor/itensor.hpp"
#include "include/tensor/tensor_actor.hpp"

#pragma once
#ifndef TENNCOR_TENSOR_HANDLER_HPP
#define TENNCOR_TENSOR_HANDLER_HPP

namespace nnet
{

using CONN_ACTOR = std::function<itens_actor*(out_wrapper<void>&,
	std::vector<in_wrapper<void> >&,tenncor::tensor_proto::tensor_t)>;

using ASSIGN_FUNC = std::function<void(void*,const void*, 
	tenncor::tensor_proto::tensor_t)>;

void default_assign (void* dest, const void* src, 
	tenncor::tensor_proto::tensor_t type);

//! Generic Tensor Handler
class itensor_handler
{
public:
	//! virtual handler interface destructor
	virtual ~itensor_handler (void) {}

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	itensor_handler* clone (void) const;

	//! clone function for copying from itensor_handler
	itensor_handler* move (void);

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler* clone_impl (void) const = 0;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler* move_impl (void) = 0;

	void* get_raw (itensor& ten) const;

	const void* get_raw (const itensor& ten) const;
};

//! Transfer Function
class actor_func : public itensor_handler
{
public:
	//! tensor handler accepts a shape manipulator and a forward transfer function
	actor_func (CONN_ACTOR make_actor);

	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	actor_func* clone (void) const;

	//! clone function for copying from itensor_handler
	actor_func* move (void);

	//! performs tensor transfer function given an input tensors
	itens_actor* operator () (itensor& out, std::vector<const itensor*>& args);

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler* move_impl (void);

private:
	CONN_ACTOR make_actor_;
};

// todo: test
//! Assignment Function
class assign_func : public itensor_handler
{
public:
	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from itensor_handler
	assign_func* clone (void) const;

	//! clone function for copying from itensor_handler
	assign_func* move (void);

	//! performs tensor transfer function given an input tensor
	void operator () (itensor& out, const itensor& arg,
		ASSIGN_FUNC f = default_assign) const;

	//! performs tensor transfer function given an input array
	// asserts that size of indata allocated chunk <= out.n_elems()
	void operator () (itensor& out, const void* indata, 
		tenncor::tensor_proto::tensor_t type,
		ASSIGN_FUNC f = default_assign) const;

protected:
	//! clone implementation for copying from itensor_handler
	virtual itensor_handler* clone_impl (void) const;

	//! move implementation for moving from itensor_handler
	virtual itensor_handler* move_impl (void);
};

//! Initializer Handler
class initializer : public itensor_handler
{
public:
	// >>>> CLONE, MOVE && COPY ASSIGNMENT <<<<
	//! clone function for copying from initializer
	initializer* clone (void) const;

	//! clone function for copying from initializer
	initializer* move (void);

	//! perform initialization
	void operator () (itensor& out);

protected:
	virtual void calc_data (void* dest, 
		tenncor::tensor_proto::tensor_t type, tensorshape outshape) = 0;
};

//! Constant Initializer
class const_init : public initializer
{
public:
	const_init (void);

	const_init (double data);

	template <typename T>
	void set (T value)
	{
		type_ = get_prototype<T>();
		value_ = nnutils::stringify(&value, 1);
	}

	//! clone function for copying from parents
	const_init* clone (void) const;

	//! clone function for copying from parents
	const_init* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler* clone_impl (void) const;

	//! move implementation for moving from parents
	virtual itensor_handler* move_impl (void);
	
	//! initialize data as constant
	virtual void calc_data (void* dest, 
		tenncor::tensor_proto::tensor_t type, tensorshape outshape);
		
private:
	std::string value_;

	tenncor::tensor_proto::tensor_t type_;
};

//! Uniformly Random Initializer
class rand_uniform : public initializer
{
public:
	//! initialize tensors with a random value between min and max
	rand_uniform (double min, double max);

	//! clone function for copying from parents
	rand_uniform* clone (void) const;

	//! clone function for copying from parents
	rand_uniform* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler* clone_impl (void) const;

	//! clone function for copying from parents
	virtual itensor_handler* move_impl (void);
	
	//! initialize data as constant
	virtual void calc_data (void* dest, 
		tenncor::tensor_proto::tensor_t type, tensorshape outshape);

private:
	std::uniform_real_distribution<double>  distribution_;
};

class rand_uniform_int : public initializer
{
public:
	//! initialize tensors with a random value between min and max
	rand_uniform_int (signed min, signed max);

	//! clone function for copying from parents
	rand_uniform_int* clone (void) const;

	//! clone function for copying from parents
	rand_uniform_int* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler* clone_impl (void) const;

	//! clone function for copying from parents
	virtual itensor_handler* move_impl (void);
	
	//! initialize data as constant
	virtual void calc_data (void* dest, 
		tenncor::tensor_proto::tensor_t type, tensorshape outshape);

private:
	std::uniform_int_distribution<signed>  distribution_;
};

//! Normal Random Initializer
class rand_normal : public initializer
{
public:
	//! initialize tensors with a random value between min and max
	rand_normal (double mean = 0, double stdev = 1);

	//! clone function for copying from parents
	rand_normal* clone (void) const;

	//! clone function for copying from parents
	rand_normal* move (void);

protected:
	//! clone implementation for copying from parents
	virtual itensor_handler* clone_impl (void) const;

	//! clone function for copying from parents
	virtual itensor_handler* move_impl (void);

	//! initialize data as constant
	virtual void calc_data (void* dest, 
		tenncor::tensor_proto::tensor_t type, tensorshape outshape);

private:
	std::normal_distribution<double> distribution_;
};

}

#endif /* TENNCOR_TENSOR_HANDLER_HPP */
