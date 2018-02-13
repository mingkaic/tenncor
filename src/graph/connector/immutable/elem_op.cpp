//
//  elem_op.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/elem_op.hpp"

#ifdef TENNCOR_ELEM_OP_HPP

namespace nnet
{

static inline tensorshape elementary_shaper (std::vector<tensorshape> shapes)
{
	tensorshape lastshape;
	for (size_t i = 0, nshapes = shapes.size(); i < nshapes; ++i)
	{
		if (shapes[i].n_elems() == 1)
		{
			continue;
		}
		if (false == shapes[i].is_compatible_with(lastshape))
		{
			std::stringstream ss;
			ss << "shape ";
			print_shape(shapes[i], ss);
			ss << " is incompatible with shape ";
			print_shape(lastshape, ss);
			throw std::runtime_error(ss.str());
		}
		lastshape = shapes[i];
	}
	if (false == lastshape.is_fully_defined()) return std::vector<size_t>{1};
	return lastshape;
}

static std::unordered_set<std::string> opset;

elem_op* elem_op::get (std::vector<inode*> args, std::string opname, BACK_MAP bwd)
{
	if (opset.empty()) opset = all_ops();
	assert(false == args.empty() && opset.end() != opset.find(opname));
	return new elem_op(args, opname, bwd);
}

elem_op* elem_op::get (std::vector<inode*> args, tensorshape shape, std::string opname, BACK_MAP bwd)
{
	if (opset.empty()) opset = all_ops();
	assert(false == args.empty() && opset.end() != opset.find(opname));
	return new elem_op(args, shape, opname, bwd);
}

elem_op* elem_op::clone (void) const
{
	return static_cast<elem_op*>(clone_impl());
}

elem_op* elem_op::move (void)
{
	return static_cast<elem_op*>(move_impl());
}

elem_op& elem_op::operator = (const elem_op& other)
{
	if (this != &other)
	{
		immutable::operator = (other);
		copy_helper(other);
	}
	return *this;
}

elem_op& elem_op::operator = (elem_op&& other)
{
	if (this != &other)
	{
		immutable::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}


elem_op::elem_op (std::vector<inode*> args, 
	std::string opname, BACK_MAP bwd) :
immutable(args, opname), fwd_(
[opname](std::vector<const tensor*> srcs) -> tensor*
{
	std::vector<tensorshape> srcshapes(srcs.size());
	std::transform(srcs.begin(), srcs.end(), srcshapes.begin(), 
	[](const tensor* src) { return src->get_shape(); });
	// type convert source
	std::shared_ptr<operate_io> io = std::make_shared<operate_io>(opname);
	for (size_t i = 0; i < srcs.size(); ++i)
	{
		srcs[i]->write_to(*io, i);
	}
	return new tensor(elementary_shaper(srcshapes), io);
}), bwd_(bwd) { this->update(); }

elem_op::elem_op (std::vector<inode*> args, 
	tensorshape shape, std::string opname, BACK_MAP bwd) :
immutable(args, opname), fwd_(
[opname, shape](std::vector<const tensor*> srcs) -> tensor*
{
	std::vector<tensorshape> srcshapes(srcs.size());
	std::transform(srcs.begin(), srcs.end(), srcshapes.begin(), 
	[](const tensor* src) { return src->get_shape(); });
	// type convert source
	std::shared_ptr<operate_io> io = std::make_shared<operate_io>(opname);
	for (size_t i = 0; i < srcs.size(); ++i)
	{
		srcs[i]->write_to(*io, i);
	}
	return new tensor(shape, io);
}), bwd_(bwd) { this->update(); }

elem_op::elem_op (const elem_op& other) :
	immutable(other)
{
	copy_helper(other);
}

elem_op::elem_op (elem_op&& other) :
	immutable(std::move(other))
{
	move_helper(std::move(other));
}

inode* elem_op::clone_impl (void) const
{
	return new elem_op(*this);
}

inode* elem_op::move_impl (void)
{
	return new elem_op(std::move(*this));
}

void elem_op::forward_pass (std::vector<inode*>& args)
{
	if (nullptr == data_)
	{
		std::vector<const tensor*> tens(args.size());
		std::transform(args.begin(), args.end(), tens.begin(),
		[](inode* arg)
		{
			tensor* ten = arg->get_tensor();
			if (nullptr == ten)
			{
				throw std::exception(); // todo: better exception
			}
			return ten;
		});
		// assert none of tens is null
		data_ = std::unique_ptr<tensor>(fwd_(tens));
	}
	data_->copy();
}

varptr elem_op::backward_pass (inode* wrt)
{
	std::vector<inode*> args = this->get_arguments();
	std::vector<std::pair<inode*,inode*> > deps;
	for (inode* arg : args)
	{
		deps.push_back({arg, arg->derive(wrt)});
	}
	return bwd_(deps);
}


void elem_op::copy_helper (const elem_op& other)
{
	shaper_ = other.shaper_;
	fwd_ = other.fwd_;
	bwd_ = other.bwd_;
}

void elem_op::move_helper (elem_op&& other)
{
	shaper_ = std::move(other.shaper_);
	fwd_ = std::move(other.fwd_);
	bwd_ = std::move(other.bwd_);
}

}

#endif
