
// static Identifier* flatten_mat (Identifier* a)
// {
// 	return coord_func({a},
// 	[](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
// 	{
// 		assert(srcs.size() == 1);
// 		size_t per = type_size(type);
// 		char* out = (char*) dest.first;
// 		const char* in = (const char*) srcs[0].first;
// 		tshape inshape = srcs[0].second;
// 		size_t nelems = inshape.n_elems();
// 		// zero out non-trace
// 		std::memset(out, 0, per * nelems * nelems);
// 		for (size_t i = 0; i < nelems; i++)
// 		{
// 			size_t outidx = i * nelems + i; // populate the trace
// 			std::memcpy(out + outidx * per, in + i * per, per);
// 		}
// 	},
// 	[](tshape inshape, std::vector<uint64_t>) -> tshape
// 	{
// 		size_t nelems = inshape.n_elems();
// 		return std::vector<size_t>{nelems, nelems};
// 	}, INJACOBIAN);
// }

// Identifier* matmul_grad (Identifier* x, std::vector<Identifier*> args)
// {
// 	Identifier* a = args.front();
// 	Identifier* b = args.back();
// 	Identifier* adx = a->derive(x);
// 	Identifier* bdx = b->derive(x);
// 	Identifier* c = matmul(a, b);
// 	// process both arguments as jacobians
// 	Identifier* da, db;
// 	tshape bases = x->get_tensor()->get_shape();
// 	if (nullptr != adx.get())
// 	{
// 		adx = flatten_mat(adx); // todo: ensure adx keeps jacobian shape <xshape, cshape>
// 		// todo: check for parent ja
// 		Identifier* ja = functor::get({b, a, c},
// 		[](std::unique_ptr<idata_src>& src, std::vector<Identifier*> args)
// 		{
// 			Identifier* b = args[0];
// 			Identifier* a = args[1];
// 			Identifier* c = args[2];
// 			operate_io* csrc = new operate_io(
// 			[](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
// 			{
// 				assert(srcs.size() == 1);
// 				size_t per = type_size(type);
// 				char* out = (char*) dest.first;
// 				const char* in = (const char*) srcs[0].first;
// 				tshape outshape = dest.second;
// 				tshape inshape = srcs[0].second;
// 				size_t xlimit = inshape[0];
// 				size_t ylimit = inshape[1];
// 				size_t ns = inshape.n_elems();
// 				size_t nparts = outshape[0] / xlimit;
// 				// zero out background
// 				std::memset(out, 0, outshape.n_elems() * per);
// 				for (size_t i = 0; i < ns; ++i)
// 				{
// 					std::vector<size_t> coord = inshape.coord_from_idx(i);
// 					size_t x = coord[0];
// 					size_t y = coord[1];
// 					for (size_t j = 0; j < nparts; ++j)
// 					{
// 						coord[0] = x + j * xlimit;
// 						coord[1] = y + j * ylimit;
// 						std::memcpy(out + outshape.flat_idx(coord) * per, in + i * per, per);
// 					}
// 				}
// 			});
// 			src = std::unique_ptr<idata_src>(csrc);
// 			size_t nouter = a->get_tensor()->get_shape().n_elems();
// 			size_t ninner = c->get_tensor()->get_shape().n_elems();
// 			tshape outshape({ninner, nouter});
// 			b->get_tensor()->write_to(*csrc);
// 			return new tensor(outshape);
// 		},
// 		[](Identifier*, std::vector<Identifier*>) -> Identifier*
// 		{
// 			throw std::bad_function_call(); // unimplemented
// 		}, JACOBIANLEFT); // ja has shape <ashape, cshape>
// 		da = matmul(adx, ja); // da should have shape <xshape, ashape>
// 	}
// 	if (nullptr != bdx.get())
// 	{
// 		bdx = flatten_mat(bdx);
// 		Identifier* jb = functor::get({a, b, c},
// 		[](std::unique_ptr<idata_src>& src, std::vector<Identifier*> args)
// 		{
// 			Identifier* a = args[0];
// 			Identifier* b = args[1];
// 			Identifier* c = args[2];
// 			operate_io* csrc = new operate_io(
// 			[](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
// 			{
// 				assert(srcs.size() == 2);
// 				size_t per = type_size(type);
// 				char* out = (char*) dest.first;
// 				const char* in = (const char*) srcs[0].first;
// 				tshape outshape = dest.second;
// 				tshape inshape = srcs[0].second;
// 				std::vector<size_t> blist = srcs[1].second.as_list();
// 				size_t ylimit = inshape[0];
// 				size_t xlimit = blist[0];
// 				size_t ns = inshape.n_elems();
// 				// zero out background
// 				std::memset(out, 0, outshape.n_elems() * per);
// 				for (size_t i = 0; i < ns; ++i)
// 				{
// 					std::vector<size_t> coord = inshape.coord_from_idx(i);
// 					size_t x = coord[1] * ylimit;
// 					size_t y = coord[0] * xlimit;
// 					for (size_t j = 0; j < xlimit; ++j)
// 					{
// 						coord[0] = x + j;
// 						coord[1] = y + j;
// 						std::memcpy(out + outshape.flat_idx(coord) * per, in + i * per, per);
// 					}
// 				}
// 			});
// 			src = std::unique_ptr<idata_src>(csrc);
// 			tshape bshape = b->get_tensor()->get_shape();
// 			size_t nouter = bshape.n_elems();
// 			size_t ninner = c->get_tensor()->get_shape().n_elems();
// 			tshape outshape({ninner, nouter});
// 			a->get_tensor()->write_to(*csrc, 0);
// 			b->get_tensor()->write_to(*csrc, 1);
// 			return new tensor(outshape);
// 		},
// 		[](Identifier*, std::vector<Identifier*>) -> Identifier*
// 		{
// 			throw std::bad_function_call(); // unimplemented
// 		}, JACOBIANRIGHT);
// 		db = matmul(bdx, jb);
// 	}
// 	// todo: ensure non-terminating matmul keep jacobian shape
// 	return coord_func({reduce_sum(da + db, 0)},
// 	[](TENS_TYPE type, VARR_T dest, std::vector<CVAR_T> srcs)
// 	{
// 		assert(srcs.size() == 1);
// 		size_t per = type_size(type);
// 		char* out = (char*) dest.first;
// 		const char* in = (const char*) srcs[0].first;
// 		tshape outshape = dest.second;
// 		size_t n = outshape.n_elems();
// 		std::memcpy(out, in, n * per);
// 	},
// 	[bases](tshape, std::vector<uint64_t>)
// 	{
// 		return bases;
// 	}, OUTJACOBIAN);
// }
