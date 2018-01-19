//
//  immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/immutable.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

immutable::~immutable (void)
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

immutable* immutable::get (std::vector<inode*> args,
	SHAPER shaper, actor_func* Nf,
	BACK_MAP ginit, std::string name,
	inode* ignore_jacobian)
{
	assert(false == args.empty());
	immutable* imm = new immutable(args, shaper, Nf, ginit, name);
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

immutable* immutable::clone (void) const
{
	return static_cast<immutable*>(this->clone_impl());
}

immutable* immutable::move (void)
{
	return static_cast<immutable*>(this->move_impl());
}

immutable& immutable::operator = (const immutable& other)
{
	if (this != &other)
	{
		base_immutable::operator = (other);
		copy_helper(other);
	}
	return *this;
}

immutable& immutable::operator = (immutable&& other)
{
	if (this != &other)
	{
		base_immutable::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

immutable::immutable (
	std::vector<inode*> args,
	SHAPER shaper,
	actor_func* Nf,
	BACK_MAP ginit, std::string label) :
base_immutable(args, label),
shaper_(shaper), Nf_(Nf),
ginit_(ginit) { this->update(std::unordered_set<size_t>{}); }

immutable::immutable (const immutable& other) :
	base_immutable(other)
{
	copy_helper(other);
}

immutable::immutable (immutable&& other) :
	base_immutable(std::move(other))
{
	move_helper(std::move(other));
}

inode* immutable::clone_impl (void) const
{
	return new immutable(*this);
}

inode* immutable::move_impl (void)
{
	return new immutable(std::move(*this));
}

base_immutable* immutable::arg_clone (std::vector<inode*> args) const
{
	if (nullptr == Nf_)
	{
		throw std::exception(); // todo: better exception
	}
	return new immutable(args, shaper_, Nf_->clone(), ginit_, this->get_label());
}

void immutable::forward_pass (void)
{
	if (nullptr == actor_)
	{
		// shape and tensor extraction
		std::vector<const itensor*> tens;
		std::vector<tensorshape> ts;
		std::vector<tenncor::tensor_proto::tensor_t> types;
		// todo: determine whether or not to move this tensor extraction up to base_immutable::update
		for (subject* sub : this->dependencies_)
		{
			const itensor* arg = this->take_eval(static_cast<inode*>(sub));
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
			tenncor::tensor_proto::tensor_t type = types.size() ? 
				types[0] : tenncor::tensor_proto::DOUBLE_T;
			switch (type)
			{
				case tenncor::tensor_proto::DOUBLE_T:
					this->data_ = new tensor_double(s);
				break;
				case tenncor::tensor_proto::SIGNED_T:
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

void immutable::backward_pass (variable* leaf)
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

void immutable::copy_helper (const immutable& other)
{
	if (Nf_) delete Nf_;

	ginit_ = other.ginit_;
	Nf_ = other.Nf_->clone();
	shaper_ = other.shaper_;
}

void immutable::move_helper (immutable&& other)
{
	if (Nf_) delete Nf_;

	ginit_ = std::move(other.ginit_);
	Nf_ = other.Nf_->move();
	shaper_ = std::move(other.shaper_);
}

}

#endif
