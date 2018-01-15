//
//  const_immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/const_immutable.hpp"
#include "include/graph/leaf/constant.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

const_immutable* const_immutable::get (inode* x)
{
	return new const_immutable(x);
}


const_immutable::const_immutable (inode* x) :
	immutable(std::vector<inode*>{x},
	[](std::vector<tensorshape> shapes) { return shapes[0]; },
	new transfer_func<double>([](double* dest, std::vector<const double*> src, shape_io shape)
	{
		size_t n_elems = shape.outs_.n_elems();
		std::memcpy(dest, src[0], sizeof(double) * n_elems);
	}),
	[](std::vector<std::pair<inode*,inode*>>)
	{
		return constant::get_shared_zero();
	}, "const_immutable")
{
	this->gcache_.clear();
	this->jacobians_.clear();
}

}

#endif
