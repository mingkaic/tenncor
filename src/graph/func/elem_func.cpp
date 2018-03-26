//
//  elem_op.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/func/elem_func.hpp"

#ifdef TENNCOR_ELEM_FUNC_HPP

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

functor* elem_func (std::vector<inode*> args, std::string opname, OPCODE op, BACKMAP_F bwd)
{
	assert(has_ele(opname));
	return functor::get(args,
	[opname](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		operate_io* io = new operate_io(ebind_name(opname));
		src = std::unique_ptr<idata_src>(io);
		std::vector<tensorshape> srcshapes;
		for (size_t i = 0; i < args.size(); ++i)
		{
			tensor* tens = args[i]->get_tensor();
			if (nullptr == tens)
			{
				throw std::exception(); // todo: better exception
			}
			srcshapes.push_back(tens->get_shape());
			tens->write_to(*io, i);
		}
		// invariant: none of tens is null
		return new tensor(elementary_shaper(srcshapes));
	},
	[bwd](inode* wrt, std::vector<inode*> args)
	{
		std::vector<std::pair<inode*,inode*> > deps;
		for (inode* arg : args)
		{
			deps.push_back({arg, arg->derive(wrt)});
		}
		return bwd(deps);
	}, op);
}

}

#endif
