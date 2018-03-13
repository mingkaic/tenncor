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

functor* coord_func (inode* arg, SIDX_F smap, USHAPE_F shaper, std::string name)
{
	return functor::get({arg},
	[smap, shaper](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		sindex_io* sio = new sindex_io(smap);
		src = std::unique_ptr<idata_src>(sio);
		tensor* tens = args[0]->get_tensor();
		assert(nullptr != tens);
		// assert that shape only change once
		tens->write_to(*sio);
		return new tensor(shaper(tens->get_shape()));
	},
	[](inode* wrt, std::vector<inode*> args)
	{
		return args[0]->derive(wrt);
	}, name);
}

}

#endif
