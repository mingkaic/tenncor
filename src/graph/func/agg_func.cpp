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

functor* agg_func (inode* arg, std::string opname, OPCODE op, BACKMAP_F bwd)
{
	assert(has_agg(opname));
	return functor::get({arg},
	[opname](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		assert(args.size() == 1);
		operate_io* asrc = new operate_io(
		[opname](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
		{
			assert(srcs.size() == 1);
			char* out = (char*) dest.first;
			const char* in = (const char*) srcs[0].first;
			tensorshape outshape = dest.second;
			tensorshape inshape = srcs[0].second;
			std::vector<size_t> index(inshape.n_elems(), 0);
		
			std::unordered_set<size_t> outmap;
			size_t per = type_size(type);
			AFUNC_F agg = abind_name(opname)(type);
			for (size_t i = 0; i < index.size(); ++i)
			{
				if (outmap.end() == outmap.find(index[i]))
				{
					memcpy(out + index[i] * per, in + i * per, per);
					outmap.insert(index[i]);
				}
				else
				{
					agg(i, out + index[i] * per, (void*) in);
				}
			}
		});
		src = std::unique_ptr<idata_src>(asrc);
		const tensor* tens = args[0]->get_tensor();
		assert(tens && tens->has_data());
		tens->write_to(*asrc);
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
	}, op);
}

functor* agg_func (inode* arg, inode* dimension, std::string opname, OPCODE op, BACKMAP_F bwd)
{
	assert(has_agg(opname));
	return functor::get({arg, dimension},
	[opname](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		assert(args.size() == 2);
		operate_io* asrc = new operate_io(
		[opname](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
		{
			assert(srcs.size() == 2);
			char* out = (char*) dest.first;
			const char* in = (const char*) srcs[0].first;
			uint64_t dim = *((uint64_t*) srcs[1].first);
			tensorshape outshape = dest.second;
			tensorshape inshape = srcs[0].second;

			size_t rank = inshape.rank();
			assert(rank > dim);
			size_t nin = inshape.n_elems();
			std::vector<size_t> index(nin, 0);
			if (rank > 1)
			{
				std::vector<size_t> coord;
				for (size_t i = 0; i < nin; ++i)
				{
					coord = inshape.coord_from_idx(i);
					coord.erase(coord.begin() + dim);
					index[i] = outshape.flat_idx(coord);
				}
			}

			std::unordered_set<size_t> outmap;
			size_t per = type_size(type);
			AFUNC_F agg = abind_name(opname)(type);
			for (size_t i = 0; i < index.size(); ++i)
			{
				if (outmap.end() == outmap.find(index[i]))
				{
					memcpy(out + index[i] * per, in + i * per, per);
					outmap.insert(index[i]);
				}
				else
				{
					agg(i, out + index[i] * per, (void*) in);
				}
			}
		},
		[](std::vector<TENS_TYPE> types)
		{
			assert(types.size() == 2 &&
				DOUBLE != types[1] && FLOAT != types[1]);
			return types[0];
		});
		src = std::unique_ptr<idata_src>(asrc);
		inode* arg = args[0];
		inode* darg = args[1];
		tensor* tens = arg->get_tensor();
		tensor* dtens = darg->get_tensor();
		tensorshape shape = tens->get_shape();
		uint64_t dim = expose<uint64_t>(args[1])[0];
		assert(tens && dtens && shape.rank() > dim);
		// assert that shape only change once
		tens->write_to(*asrc, 0);
		dtens->write_to(*asrc, 1);
		std::vector<size_t> slist = shape.as_list();
		if (1 == slist.size())
		{
			slist[0] = 1;
		}
		else
		{
			slist.erase(slist.begin() + dim);
		}
		return new tensor(tensorshape(slist));
	},
	[bwd](inode* wrt, std::vector<inode*> args)
	{
		inode* arg = args[0];
		inode* shapeinfo = args[1];
		return bwd({{arg, arg->derive(wrt)}, {shapeinfo, nullptr}});
	}, op);
}

}

#endif
