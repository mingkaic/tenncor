//
// Created by Mingkai Chen on 2017-07-03.
//

#include "include/graph/connector/immutable/shape_dep.hpp"

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

shape_dep* shape_dep::get (inode* arg, USIDX_F extracter,
	USHAPE_F shaper, std::string name)
{
	return new shape_dep(arg, extracter, shaper, name);
}

shape_dep* shape_dep::clone (void) const
{
	return static_cast<shape_dep*>(this->clone_impl());
}

shape_dep* shape_dep::move (void)
{
	return static_cast<shape_dep*>(this->move_impl());
}

shape_dep& shape_dep::operator = (const shape_dep& other)
{
	if (this != &other)
	{
		immutable::operator = (other);
		copy_helper(other);
	}
	return *this;
}

shape_dep& shape_dep::operator = (shape_dep&& other)
{
	if (this != &other)
	{
		immutable::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}


shape_dep::shape_dep (inode* arg, USIDX_F extracter,
	USHAPE_F shaper, std::string label) :
immutable({arg}, label), extracter_(extracter), shaper_(shaper)
{
	this->update();
}

shape_dep::shape_dep (const shape_dep& other) :
	immutable(other)
{
	copy_helper(other);
}

shape_dep::shape_dep (shape_dep&& other) :
	immutable(std::move(other))
{
	move_helper(std::move(other));
}

inode* shape_dep::clone_impl (void) const
{
	return new shape_dep(*this);
}

inode* shape_dep::move_impl (void)
{
	return new shape_dep(std::move(*this));
}

void shape_dep::forward_pass (std::vector<inode*>& args)
{
	if (tensor* tens = args[0]->get_tensor())
	{
		tensorshape shape = tens->get_shape();
		if (nullptr == data_)
		{
			data_ = std::make_unique<tensor>(shaper_(shape));
		}

		std::vector<size_t> sdata = extracter_(shape);
		size_t ns = sdata.size();
		std::vector<double> doub_d(sdata.begin(), sdata.end());
		std::shared_ptr<void> ptr = shared_varr(ns);
		std::memcpy(ptr.get(), &doub_d[0], sizeof(double) * ns);
		asgn_.set_data(ptr, DOUBLE, data_->get_shape(), 0); // todo: make tens's type
		data_->read_from(asgn_);
	}
}

varptr shape_dep::backward_pass (inode* wrt)
{
	varptr out;
	if (this == wrt)
	{
		tensor* ten = get_tensor();
		assert(ten && ten->has_data());
		tensorshape shape = ten->get_shape();
		std::vector<double> data(shape.n_elems(), 1);
		out = constant::get(data, shape);
	}
	return out;
}


void shape_dep::copy_helper (const shape_dep& other)
{
	extracter_ = other.extracter_;
	shaper_ = other.shaper_;
}

void shape_dep::move_helper (shape_dep&& other)
{
	extracter_ = std::move(other.extracter_);
	shaper_ = std::move(other.shaper_);
}

}

#endif
