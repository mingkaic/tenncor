//
//  common.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/operate/common.hpp"

#ifdef TENNCOR_COM_FUNC_HPP

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
		operate_io* io = new operate_io(ebind(opname));
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

functor* shape_func (std::vector<inode*> args, USIDX_F extracter, USHAPE_F shaper, OPCODE op)
{
	return functor::get(args,
	[extracter, shaper](std::unique_ptr<idata_src>& src, std::vector<inode*> args) -> tensor*
	{
		tensor* tens = args[0]->get_tensor();
		assert(nullptr != tens);
		const_init* ci = new const_init();
		src = std::unique_ptr<idata_src>(ci);
		std::vector<uint64_t> sinfo;
		if (args.size() > 1)
		{
			sinfo = expose<uint64_t>(args[1]);
		}

		tensorshape shape = tens->get_shape();
		std::vector<size_t> sdata = extracter(shape, sinfo);
		std::vector<double> doub_d(sdata.begin(), sdata.end()); // todo: make tens's type
		ci->set<double>(doub_d);
		return new tensor(shaper(shape, sinfo));
	},
	[](inode*, std::vector<inode*>) -> varptr
	{
		return nullptr;
	}, op);
}

}

#endif
