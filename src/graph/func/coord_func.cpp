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

functor* coord_func (std::vector<inode*> args, VTFUNC_F cf, USHAPE_F shaper, OPCODE op)
{
	return functor::get(args,
	[cf, shaper](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		operate_io* csrc = new operate_io(cf,
		[](std::vector<TENS_TYPE> types)
		{
			// if (types.size() > 1)
			// {
			// 	assert(std::all_of(types.begin() + 1, types.end(),
			// 	[](TENS_TYPE type)
			// 	{
			// 		return DOUBLE != type && FLOAT != type;
			// 	}));
			// }
			return types[0];
		});
		src = std::unique_ptr<idata_src>(csrc);
		// assert that shape only change once
		assert(args.size() > 0);
		for (size_t i = 0; i < args.size(); ++i)
		{
			tensor* tens = args[i]->get_tensor();
			assert(nullptr != tens);
			tens->write_to(*csrc, i);
		}
		std::vector<uint64_t> sinfo;
		for (size_t i = 1; i < args.size(); ++i)
		{
			auto vec = expose<uint64_t>(args[i]);
			sinfo.insert(sinfo.end(), vec.begin(), vec.end());
		}
		return new tensor(shaper(args[0]->get_tensor()->get_shape(), sinfo));
	},
	[](inode* wrt, std::vector<inode*> args)
	{
		return args.front()->derive(wrt);
	}, op);
}

}

#endif
