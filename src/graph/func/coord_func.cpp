//
//  coord_func.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/func/coord_func.hpp"

#ifdef TENNCOR_COORD_FUNC_HPP

namespace nnet
{

functor* coord_func (std::vector<inode*> args, SIDX_F smap, USHAPE_F shaper, OPCODE op)
{
	return functor::get(args,
	[smap, shaper](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		sindex_io* sio = new sindex_io(smap);
		src = std::unique_ptr<idata_src>(sio);
		tensor* tens = args.front()->get_tensor();
		assert(nullptr != tens);
		// assert that shape only change once
		tens->write_to(*sio);
		std::vector<uint64_t> sinfo;
		if (args.size() > 1)
		{
			sinfo = expose<uint64_t>(args[1]);
			sio->shape_info(sinfo);
		}
		return new tensor(shaper(tens->get_shape(), sinfo));
	},
	[](inode* wrt, std::vector<inode*> args)
	{
		return args.front()->derive(wrt);
	}, op);
}

}

#endif
