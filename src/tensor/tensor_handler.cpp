//
//  tensor_handler.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/tensor/tensor_handler.hpp"

#ifdef TENNCOR_TENSOR_HANDLER_HPP

#include "include/utils/utils.hpp"

namespace nnet
{

void default_assign (void* dest, const void* src, 
	tenncor::tensor_proto::tensor_t type)
{
	memcpy(dest, src, type_size(type));
}

// INTERFACE

itensor_handler* itensor_handler::clone (void) const
{
	return clone_impl();
}

itensor_handler* itensor_handler::move (void)
{
	return move_impl();
}

void* itensor_handler::get_raw (itensor& ten) const
{
	assert(ten.is_alloc());
	return ten.get_data();
}

const void* itensor_handler::get_raw (const itensor& ten) const
{
	assert(ten.is_alloc());
	return ten.get_data();
}

// ACTOR CREATION FUNCTION

actor_func::actor_func (CONN_ACTOR make_actor) : make_actor_(make_actor) {}

actor_func* actor_func::clone (void) const
{
	return static_cast<actor_func*>(clone_impl());
}

actor_func* actor_func::move (void)
{
	return static_cast<actor_func*>(move_impl());
}

itens_actor* actor_func::operator () (itensor& out, std::vector<const itensor*>& args)
{
	tenncor::tensor_proto::tensor_t type = out.get_type();
	size_t n_arg = args.size();
	out_wrapper<void> dest
	{
		this->get_raw(out), 
		out.get_shape()
	};
	
	std::vector<in_wrapper<void> > sources(n_arg);
	std::transform(args.begin(), args.end(), sources.begin(),
	[this, type](const itensor* arg)
	{
		if (type != arg->get_type())
		{
			throw std::exception();
		}
		return in_wrapper<void>
		{
			this->get_raw(*arg), 
			arg->get_shape()
		};
	});
	return this->make_actor_(dest, sources, type);
}

itensor_handler* actor_func::clone_impl (void) const
{
	return new actor_func(*this);
}

itensor_handler* actor_func::move_impl (void)
{
	return new actor_func(std::move(*this));
}

// ASSIGNMENT FUNCTION

assign_func* assign_func::clone (void) const
{
	return static_cast<assign_func*>(clone_impl());
}

assign_func* assign_func::move (void)
{
	return static_cast<assign_func*>(move_impl());
}

void assign_func::operator () (itensor& out, const itensor& arg, ASSIGN_FUNC f) const
{
	tensorshape outshape = out.get_shape();
	tenncor::tensor_proto::tensor_t type = out.get_type();
	if (type != arg.get_type())
	{
		throw std::exception();
	}
	size_t n = arg.n_elems();
	assert(n == outshape.n_elems());
	char* dest = (char*) this->get_raw(out);
	const char* indata = (const char*) this->get_raw(arg);
	size_t bytesize = type_size(type);
	for (size_t i = 0; i < n; i++)
	{
		f((void*) (dest + bytesize * i), (const void*) (indata + bytesize * i), type);
	}
}

void assign_func::operator () (itensor& out, const void* indata, 
	tenncor::tensor_proto::tensor_t type, ASSIGN_FUNC f) const
{
	tensorshape outshape = out.get_shape();
	if (type != out.get_type())
	{
		throw std::exception();
	}
	size_t n = outshape.n_elems();
	char* dest = (char*) this->get_raw(out);
	const char* src = (const char*) indata;
	size_t bytesize = type_size(type);
	for (size_t i = 0; i < n; i++)
	{
		f((void*) (dest + bytesize * i), (const void*) (src + bytesize * i), type);
	}
}

itensor_handler* assign_func::clone_impl (void) const
{
	return new assign_func(*this);
}

itensor_handler* assign_func::move_impl (void)
{
	return new assign_func(std::move(*this));
}

// INITIALIZER INTERFACE

initializer* initializer::clone (void) const
{
	return static_cast<initializer*>(this->clone_impl());
}

initializer* initializer::move (void)
{
	return static_cast<initializer*>(this->move_impl());
}

void initializer::operator () (itensor& out)
{
	this->calc_data(this->get_raw(out), out.get_type(), out.get_shape());
}

// CONSTANT INITIALIZER

const_init::const_init (void) {}

const_init::const_init (double data) : 
	value_(nnutils::stringify(&data, 1)),
	type_(tenncor::tensor_proto::DOUBLE_T) {}

const_init* const_init::clone (void) const
{
	return static_cast<const_init*>(clone_impl());
}

const_init* const_init::move (void)
{
	return static_cast<const_init*>(move_impl());
}

itensor_handler* const_init::clone_impl (void) const
{
	return new const_init(*this);
}

itensor_handler* const_init::move_impl (void)
{
	return new const_init(std::move(*this));
}

void const_init::calc_data (void* dest, 
	tenncor::tensor_proto::tensor_t type, tensorshape outshape)
{
	assert(type == type_);
	size_t n = outshape.n_elems();
	size_t nbyte = value_.size();
	char* cdest = (char*) dest;
	for (size_t i = 0; i < n; ++i)
	{
		memcpy(cdest + i * nbyte, &value_[0], nbyte);
	}
}

// RANDOM INITIALIZERS

rand_uniform::rand_uniform (double min, double max) :
	distribution_(min, max) {}

rand_uniform* rand_uniform::clone (void) const
{
	return static_cast<rand_uniform*>(clone_impl());
}

rand_uniform* rand_uniform::move (void)
{
	return static_cast<rand_uniform*>(move_impl());
}

itensor_handler* rand_uniform::clone_impl (void) const
{
	return new rand_uniform(*this);
}

itensor_handler* rand_uniform::move_impl (void)
{
	return new rand_uniform(std::move(*this));
}

void rand_uniform::calc_data (void* dest, 
	tenncor::tensor_proto::tensor_t type, tensorshape outshape)
{
	assert(type == tenncor::tensor_proto::DOUBLE_T);
	size_t len = outshape.n_elems();
	double* dest_d = (double*) dest;
	auto gen = std::bind(distribution_, nnutils::get_generator());
	std::generate(dest_d, dest_d + len, gen);
}

rand_uniform_int::rand_uniform_int (signed min, signed max) :
	distribution_(min, max) {}

rand_uniform_int* rand_uniform_int::clone (void) const
{
	return static_cast<rand_uniform_int*>(clone_impl());
}

rand_uniform_int* rand_uniform_int::move (void)
{
	return static_cast<rand_uniform_int*>(move_impl());
}

itensor_handler* rand_uniform_int::clone_impl (void) const
{
	return new rand_uniform_int(*this);
}

itensor_handler* rand_uniform_int::move_impl (void)
{
	return new rand_uniform_int(std::move(*this));
}

void rand_uniform_int::calc_data (void* dest, 
	tenncor::tensor_proto::tensor_t type, tensorshape outshape)
{
	assert(type == tenncor::tensor_proto::SIGNED_T);
	signed* dest_i = (signed*) dest;
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, nnutils::get_generator());
	std::generate(dest_i, dest_i + len, gen);
}

rand_normal::rand_normal (double mean, double stdev) :
	distribution_(mean, stdev) {}

rand_normal* rand_normal::clone (void) const
{
	return static_cast<rand_normal*>(clone_impl());
}

rand_normal* rand_normal::move (void)
{
	return static_cast<rand_normal*>(move_impl());
}

itensor_handler* rand_normal::clone_impl (void) const
{
	return new rand_normal(*this);
}

itensor_handler* rand_normal::move_impl (void)
{
	return new rand_normal(std::move(*this));
}

void rand_normal::calc_data (void* dest, 
	tenncor::tensor_proto::tensor_t type, tensorshape outshape)
{
	size_t len = outshape.n_elems();
	auto gen = std::bind(distribution_, nnutils::get_generator());
	switch (type)
	{
		case tenncor::tensor_proto::DOUBLE_T:
		{
			double* dest_d = (double*) dest;
			std::generate(dest_d, dest_d + len, gen);
		}
		break;
		case tenncor::tensor_proto::SIGNED_T:
		{
			signed* dest_i = (signed*) dest;
			std::generate(dest_i, dest_i + len, gen);
		}
		break;
		default:
			throw std::exception();
	}
}

}

#endif