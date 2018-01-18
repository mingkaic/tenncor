/*!
 *
 *  tensor_actor.hpp
 *  cnnet
 *
 *  Purpose:
 *  actor abstract performs operation on input and output data
 *
 *  Created by Mingkai Chen on 2018-01-16.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_TENSOR_ACTOR_HPP
#define TENNCOR_TENSOR_ACTOR_HPP

namespace nnet
{

template <typename T>
using out_wrapper = std::pair<T*,tensorshape>;

template <typename T>
using in_wrapper = std::pair<const T*,tensorshape>;

template <typename T>
using UNI_TRANS = std::function<T(T)>;

template <typename T>
using BI_TRANS = std::function<T(T,T)>;

template <typename T>
using UNI_COMP = std::function<bool(T)>;

template <typename T>
using COMPARE = std::function<bool(T,T)>;

template <typename T>
using GEN_TRANS = std::function<void(out_wrapper<T>,std::vector<in_wrapper<T> >)>;

template <typename T>
using REDUCE = std::function<double(std::vector<double>)>;

class itens_actor
{
public:
	virtual ~itens_actor (void) {}

	virtual void action (void) = 0;
};

template <typename T>
class tens_template : public itens_actor
{
public:
	virtual ~tens_template (void) {}

protected:
	tens_template (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs) :
		dest_(out_wrapper<T>{(T*) dest.first, dest.second}),
		srcs_(srcs.size())
	{
		std::transform(srcs.begin(), srcs.end(), srcs_.begin(),
		[](in_wrapper<void>& src)
		{
			return in_wrapper<T>{(T*) src.first, src.second};
		});
	}

	out_wrapper<T> dest_;

	std::vector<in_wrapper<T> > srcs_;
};

template <typename T>
class tens_elem_uni : public tens_template<T>
{
public:
	tens_elem_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		UNI_TRANS<T> trans);

	virtual void action (void);

private:
	UNI_TRANS<T> trans_;
};

template <typename T>
class tens_elems_bi : public tens_template<T>
{
public:
	tens_elems_bi (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		BI_TRANS<T> trans);

	virtual void action (void);

private:
	BI_TRANS<T> trans_;
};

template <typename T>
class tens_axial_elems : public tens_template<T>
{
public:
	tens_axial_elems (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		size_t axis, bool left, BI_TRANS<T> trans);

	virtual void action (void);

private:
	BI_TRANS<T> trans_;

	size_t axis_;

	short idx_;
};

template <typename T>
class tens_conditional_uni : public tens_template<T>
{
public:
	tens_conditional_uni (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		UNI_COMP<T> comp);

	virtual void action (void);

private:
	UNI_COMP<T> comp_;
};

template <typename T>
class tens_conditional : public tens_template<T>
{
public:
	tens_conditional (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		COMPARE<T> comp);

	virtual void action (void);

private:
	COMPARE<T> comp_;
};

template <typename T>
class tens_general : public tens_template<T>
{
public:
	tens_general (out_wrapper<void> dest, 
		std::vector<in_wrapper<void> > srcs,
		GEN_TRANS<T> trans);

	virtual void action (void);

private:
	GEN_TRANS<T> trans_;
};

}

#include "src/tensor/tensor_actor.ipp"

#endif /* TENNCOR_TENSOR_ACTOR_HPP */
