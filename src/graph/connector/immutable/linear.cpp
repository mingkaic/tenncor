//
//  linear.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/linear.hpp"

#ifdef TENNCOR_LINEAR_HPP

namespace nnet
{

linear::~linear (void)
{
	if (Nf_)
	{
		delete Nf_;
	}

	if (actor_)
	{
		delete actor_;
	}
}

linear* linear::get (std::vector<inode*> args,
	SHAPER shaper, actor_func* Nf,
	BACK_MAP ginit, std::string name,
	inode* ignore_jacobian)
{
	assert(false == args.empty());
	linear* imm = new linear(args, shaper, Nf, ginit, name);
	if (nullptr != ignore_jacobian)
	{
		std::unordered_set<ileaf*> leaves = ignore_jacobian->get_leaves();
		for (ileaf* leaf : leaves)
		{
			if (variable* var = dynamic_cast<variable*>(leaf))
			{
				imm->jacobians_.erase(var);
			}
		}
	}
	return imm;
}

linear* linear::clone (void) const
{
	return static_cast<linear*>(this->clone_impl());
}

linear* linear::move (void)
{
	return static_cast<linear*>(this->move_impl());
}

linear& linear::operator = (const linear& other)
{
	if (this != &other)
	{
		immutable::operator = (other);
		copy_helper(other);
	}
	return *this;
}

linear& linear::operator = (linear&& other)
{
	if (this != &other)
	{
		immutable::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

linear::linear (
	std::vector<inode*> args,
	SHAPER shaper,
	actor_func* Nf,
	BACK_MAP ginit, std::string label) :
immutable(args, label), shaper_(shaper), Nf_(Nf),
ginit_(ginit) { this->update(); }

linear::linear (const linear& other) :
	immutable(other)
{
	copy_helper(other);
}

linear::linear (linear&& other) :
	immutable(std::move(other))
{
	move_helper(std::move(other));
}

inode* linear::clone_impl (void) const
{
	return new linear(*this);
}

inode* linear::move_impl (void)
{
	return new linear(std::move(*this));
}

immutable* linear::arg_clone (std::vector<inode*> args) const
{
	if (nullptr == Nf_)
	{
		throw std::exception(); // todo: better exception
	}
	return new linear(args, shaper_, Nf_->clone(), ginit_, this->get_label());
}

void linear::forward_pass (void)
{
	if (nullptr == actor_)
	{
		// shape and tensor extraction
		std::vector<const tensor*> tens;
		std::vector<tensorshape> ts;
		std::vector<TENS_TYPE> types;
		// todo: determine whether or not to move this tensor extraction up to immutable::update
		for (subject* sub : this->dependencies_)
		{
			const tensor* arg = this->take_eval(static_cast<inode*>(sub));
			if (nullptr == arg)
			{
				throw std::exception(); // todo: better exception
			}
			assert(arg->is_alloc());
			types.push_back(arg->get_type());
			ts.push_back(arg->get_shape());
			tens.push_back(arg);
		}
		// shape check and tensor initialization
		tensorshape s = shaper_(ts);
		if (nullptr == this->data_)
		{
			s.assert_is_fully_defined();
			// resolve type (todo: implement)
			TENS_TYPE type = types.size() ?
				types[0] : DOUBLE;
			switch (type)
			{
				case DOUBLE:
					this->data_ = new tensor_double(s);
				break;
				case INT:
					this->data_ = new tensor_signed(s);
				break;
				default:
					throw std::exception(); // unsupported type
			}
		}
		else if (s.is_fully_defined())
		{
			// if data_ is allocated, verify shape with data_
			if (this->data_->is_alloc())
			{
				tensorshape oshape = this->data_->get_shape();
				if (false == s.is_compatible_with(oshape))
				{
					std::stringstream ss;
					print_shape(s, ss);
					ss << " is incompatible with output shape ";
					print_shape(oshape, ss);
					throw std::runtime_error(ss.str());
				}
			}
			// otherwise allocate data_
			else
			{
				this->data_->allocate(s);
			}
		}
		// assert none of tens is null
		actor_ = (*Nf_)(*(this->data_), tens);
	}
	actor_->action();
}

void linear::backward_pass (variable* leaf)
{
	std::vector<std::pair<inode*,inode*>> deps;
	for (subject* s : this->dependencies_)
	{
		inode* fn = static_cast<inode*>(s);
		inode* bn;
		if (this->jacobians_[leaf].terminal_)
		{
			bn = fn->derive(leaf); // take jacobian
		}
		else
		{
			bn = this->take_gradient(fn, leaf);
		}
		deps.push_back({fn, bn});
	}
	this->gcache_[leaf] = ginit_(deps);
}

void linear::copy_helper (const linear& other)
{
	if (Nf_) delete Nf_;

	ginit_ = other.ginit_;
	Nf_ = other.Nf_->clone();
	shaper_ = other.shaper_;
}

void linear::move_helper (linear&& other)
{
	if (Nf_) delete Nf_;

	ginit_ = std::move(other.ginit_);
	Nf_ = other.Nf_->move();
	shaper_ = std::move(other.shaper_);
}

}

#endif
