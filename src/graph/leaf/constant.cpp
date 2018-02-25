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
	return nullptr;
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
	std::shared_ptr<idata_src> source, std::string name) :
ileaf(name), data_(new tensor(shape))
{
	shape.assert_is_fully_defined();
	data_->read_from(*source);
}

}

#endif
