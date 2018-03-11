//
//  coord_mapper.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/coord_mapper.hpp"

#ifdef TENNCOR_COORD_MAPPER_HPP

namespace nnet
{

functor* coord_func (inode* arg, SIDX_F smap, USHAPE_F shaper, std::string name, bool same_fb)
{
	BACKMAP_F bwd;
	if (same_fb)
	{
		bwd = [smap, shaper, name](std::vector<std::pair<inode*,inode*> > args) -> varptr
		{
			return coord_func(args.front().second, smap, shaper, "d_" + name, true);
		};
	}
	else
	{
		bwd = [](std::vector<std::pair<inode*,inode*> > args) -> varptr
		{
			return args.front().second;
		};
	}
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
	[bwd](inode* wrt, std::vector<inode*> args)
	{
		inode* arg = args[0];
		return bwd({{arg, arg->derive(wrt)}});
	}, name);
}

}

#endif
