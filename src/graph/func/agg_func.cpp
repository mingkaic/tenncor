//
//  muxer.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/func/agg_func.hpp"
#include "include/graph/func/elem_func.hpp"
#include "include/operate/operation_utils.hpp"

#ifdef TENNCOR_AGG_FUNC_HPP

namespace nnet
{

functor* agg_func (inode* arg, std::string opname, BACKMAP_F bwd)
{
	assert(has_agg(opname));
	return functor::get({arg},
	[opname](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		assert(args.size() == 1);
		idata_io* io = new aggreg_io(opname, 
		[](tensorshape,const tensorshape inshape)
		{
			return std::vector<size_t>(inshape.n_elems(), 0);
		});
		src = std::unique_ptr<idata_src>(io);
		const tensor* tens = args[0]->get_tensor();
		assert(tens && tens->has_data());
		tens->write_to(*io);
		// invariant: none of tens is null
		return new tensor(std::vector<size_t>{1});
	},
	[bwd](inode* wrt, std::vector<inode*> args)
	{
		std::vector<std::pair<inode*,inode*> > deps;
		for (inode* arg : args)
		{
			deps.push_back({arg, arg->derive(wrt)});
		}
		return bwd(deps);
	}, opname);
}

functor* agg_func (inode* arg, std::string opname, size_t dimension, BACKMAP_F bwd)
{
	assert(has_agg(opname));
	return functor::get({arg},
	[opname, dimension](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		aggreg_io* op = new aggreg_io(opname, 
		[dimension](tensorshape outshape, const tensorshape inshape)
		{
			size_t rank = inshape.rank();
			assert(rank > dimension);
			size_t nin = inshape.n_elems();
			if (rank == 1)
			{
				return std::vector<size_t>(nin, 0);
			}
			std::vector<size_t> index(nin);
			std::vector<size_t> coord;
			for (size_t i = 0; i < nin; ++i)
			{
				coord = inshape.coord_from_idx(i);
				coord.erase(coord.begin() + dimension);
				index[i] = outshape.flat_idx(coord);
			}
			return index;
		});
		src = std::unique_ptr<idata_src>(op);

		inode* arg = args[0];
		tensor* tens = arg->get_tensor();
		tensorshape shape = tens->get_shape();
		assert(tens && shape.rank() > dimension);
		// assert that shape only change once
		tens->write_to(*op);
		std::vector<size_t> slist = shape.as_list();
		if (1 == slist.size())
		{
			slist[0] = 1;
		}
		else
		{
			slist.erase(slist.begin() + dimension);
		}
		return new tensor(tensorshape(slist));
	},
	[bwd](inode* wrt, std::vector<inode*> args)
	{
		inode* arg = args[0];
		return bwd({{arg, arg->derive(wrt)}});
	}, nnutils::formatter() << opname << "_" << dimension);
}

}

#endif
