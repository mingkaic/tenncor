//
//  constant.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/constant.hpp"

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

tensor* constant::get_tensor (void)
{
	return data_.get();
}

varptr constant::derive (inode*)
{
	tensorshape shape = data_->get_shape();
	std::vector<double> zeroes(shape.n_elems(), 0); // todo: convert to data type
	return constant::get(zeroes, shape);
}


void constant::death_on_noparent (void)
{
	if (this->get_audience().size())
	{
		delete this;
	}
}

inode* constant::clone_impl (void) const
{
	return nullptr;
}

inode* constant::move_impl (void)
{
	return nullptr;
}

constant::constant (const tensorshape& shape, 
	std::shared_ptr<idata_source> source, std::string name) :
ileaf(name), data_(new tensor(shape, source))
{
	shape.assert_is_fully_defined();
	data_->read();
}

}

#endif
