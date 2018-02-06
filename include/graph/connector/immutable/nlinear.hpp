/*!
 *
 *  nlinear.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph non-linear connector that manages a
 *  single operator's forward and backward pass
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"
#include "include/tensor/tensor_ref.hpp"
#include <memory>

#pragma once
#ifndef TENNCOR_NLINEAR_HPP
#define TENNCOR_NLINEAR_HPP

namespace nnet
{

class nlinear : public immutable
{
public:
	virtual ~nlinear (void)
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

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for nlinear, grabs ownership of Nf
	static nlinear* get (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf, BACK_MAP ginit, std::string name,
		inode* ignore_jacobian = nullptr)
	{
		assert(false == args.empty());
		nlinear* imm = new nlinear(args, shaper, Nf, ginit, name);
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

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	nlinear* clone (void) const
	{
		return static_cast<nlinear*>(this->clone_impl());
	}

	//! move function
	nlinear* move (void)
	{
		return static_cast<nlinear*>(this->move_impl());
	}

	//! declare copy assignment to copy over transfer functions
	virtual nlinear& operator = (const nlinear& other)
	{
		if (this != &other)
		{
			immutable::operator = (other);
			copy_helper(other);
		}
		return *this;
	}

	//! declare move assignment to move over transfer functions
	virtual nlinear& operator = (nlinear&& other)
	{
		if (this != &other)
		{
			immutable::operator = (std::move(other));
			move_helper(std::move(other));
		}
		return *this;
	}

protected:
	// >>>> CONSTRUCTORS <<<<
	//! nlinear constructing an aggregate transfer function
	nlinear (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf, BACK_MAP ginit, std::string label) :
	immutable(args, label), shaper_(shaper), Nf_(Nf),
	ginit_(ginit) { this->update(std::unordered_set<size_t>{}); }

	//! declare copy constructor to copy over transfer functions
	nlinear (const nlinear& other) :
		immutable(other)
	{
		copy_helper(other);
	}

	//! declare move constructor to move over transfer functions
	nlinear (nlinear&& other) :
		immutable(std::move(other))
	{
		move_helper(std::move(other));
	}

	// >>>> POLYMORPHIC CLONERS <<<<
	//! implement clone function
	virtual inode* clone_impl (void) const
	{
		return new nlinear(*this);
	}

	//! move implementation
	virtual inode* move_impl (void)
	{
		return new nlinear(std::move(*this));
	}

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual immutable* arg_clone (std::vector<inode*> args) const
	{
		if (nullptr == Nf_)
		{
			throw std::exception(); // todo: better exception
		}
		return new linear(args, shaper_, Nf_->clone(), ginit_, this->get_label());
	}

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void)
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
			this->data_ = new tensor_ref(s, tens);
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
		tensor_ref* tf = static_cast<tensor_ref*>(this->data_);
		size_t n = tf->n_elems();
		ref* refs = tf->get_refs();
		for (size_t i = 0; i < n; ++i)
		{
			ref& r = refs[i];
			std::vector<addr>& addrs = r.srcs_;
			std::vector<void*> ptrs(addrs.size(), nullptr);
			std::transform(addrs.begin(), addrs.end(), ptrs.size(),
			[](addr a)
			{
				return a.get_ptr();
			});
			tf->working_mem_[i] = fops_(ptrs);
		}
	}

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable* leaf)
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

private:
	//! copy helper
	void copy_helper (const nlinear& other)
	{
		if (Nf_) delete Nf_;

		ginit_ = other.ginit_;
		Nf_ = other.Nf_->clone();
		shaper_ = other.shaper_;
	}

	//! move helper
	void move_helper (nlinear&& other)
	{
		if (Nf_) delete Nf_;

		ginit_ = std::move(other.ginit_);
		Nf_ = other.Nf_->move();
		shaper_ = std::move(other.shaper_);
	}

	FWD_OP fops_;

	//! calculates shape of this node
	SHAPER shaper_;

	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP ginit_;
};

}

#endif /* TENNCOR_NLINEAR_HPP */
