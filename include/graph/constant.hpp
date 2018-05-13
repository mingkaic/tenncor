/*!
 *
 *  constant.hpp
 *  cnnet
 *
 *  Purpose:
 *  constant node
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/inode.hpp"

#pragma once
#ifndef TENNCOR_CONSTANT_HPP
#define TENNCOR_CONSTANT_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

class constant final : public inode
{
public:
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for scalar
	template <typename T> // todo: fix memory lost due to sole constant-functor relationship
	static varptr get (T scalar);

	//! builder for data and shape
	template <typename T>
	static varptr get (std::vector<T> raw, tshape shape);

	// static varptr get_generic (std::string data, TENS_TYPE type);

	static varptr get (tenncor::TensorPb& proto_src, std::string label);

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST SHARING) <<<<

	//! deleted copy constructor
	constant (const constant&) = delete;

	//! deleted move constructor
	constant (constant&&) = delete;

	//! copy assignment deleted
	constant& operator = (const constant&) = delete;

	//! move assignment deleted
	constant& operator = (constant&&) = delete;



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<const inode*> get_leaves (void) const
	{
		return {this};
	}

	// >>>>>> SERIALIZATION DATA <<<<<<

	virtual NODE_TYPE node_type (void) const;

	// >>>>>> SERIALIZATION ACTOR <<<<<<

	virtual void serialize_detail (google::protobuf::Any* proto_dest) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

private:
	//! name constructor, data_ is nullptr
	constant (tensor* data, std::string name);

	virtual ~constant (void);

	// >>>> POLYMORPHIC CLONERS (RETURN NULLS) <<<<

	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);


	// >>>> KILL CONDITION <<<<
	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);


	//! raw data
	TensorPtrT data_ = nullptr;
};

constant* find_const (size_t key);

bool dangling (constant* key);

void register_const (size_t key, constant* cons);

template <typename T> // todo: optimize by looking for pre-existing constants
varptr constant::get (T scalar)
{
	static_assert(std::is_arithmetic<T>::value,
		"constant must be arithmetic value");
	std::string generic((char*) &scalar, sizeof(T));
	size_t key = ((size_t) get_type<T>()) ^ std::hash<std::string>()(generic);
	constant* cons = find_const(key);
	if (nullptr == cons)
	{
		tshape shape = std::vector<size_t>{1};
		const_init ci;
		ci.set(scalar);
		tensor* data = new tensor(shape);
		data->read_from(ci);
		cons = new constant(data, nnutils::formatter() << scalar);
		register_const(key, cons);
	}
	return cons;
}

template <typename T>
varptr constant::get (std::vector<T> raw, tshape shape)
{
	static_assert(std::is_arithmetic<T>::value, 
		"constant must be arithmetic value");
	assert(shape.is_fully_defined());
	std::string name;
	if (raw.empty())
	{
		name = "<empty>";
	}
	else
	{
		name = nnutils::formatter() << raw.front() << ".." << raw.back();
	}
	const_init ci;
	ci.set(raw);
	tensor* data = new tensor(shape);
	data->read_from(ci);
	return new constant(data, name);
}

}

#endif /* TENNCOR_CONSTANT_HPP */
