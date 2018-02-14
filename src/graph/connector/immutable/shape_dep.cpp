//
// Created by Mingkai Chen on 2017-07-03.
//

#include "include/graph/connector/immutable/shape_dep.hpp"

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

shape_dep* shape_dep::get (inode* arg, SHAPE2ARR_F extracter,
	tensorshape shape, std::string name)
{
	return new shape_dep(arg, extracter, shape, name);
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


shape_dep::shape_dep (inode* arg, SHAPE2ARR_F extracter,
	tensorshape shape, std::string label) :
immutable({arg}, label), extracter_(extracter)
{
	shape.assert_is_fully_defined();
	this->data_ = std::make_unique<tensor>(shape);
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
	tensor* tens = args[0]->get_tensor();
	if (tens)
	{
		tensorshape shape = tens->get_shape();
		std::vector<size_t> sdata = extracter_(shape);
		std::vector<double> doub_d(sdata.begin(), sdata.end());
		std::shared_ptr<void> ptr = std::shared_ptr<void>(&doub_d[0]);
		asgn_.set_data(ptr, DOUBLE, data_->get_shape(), 0); // todo: make tens's type
		data_->read_from(asgn_);
	}
}

varptr shape_dep::backward_pass (inode* wrt)
{
	tensorshape shape = this->get_tensor()->get_shape();
	std::vector<double> data(shape.n_elems(),
		(double) (this == wrt));
	return constant::get(data, shape);
}


void shape_dep::copy_helper (const shape_dep& other)
{
	extracter_ = other.extracter_;
}

void shape_dep::move_helper (shape_dep&& other)
{
	extracter_ = std::move(other.extracter_);
}

}

#endif
