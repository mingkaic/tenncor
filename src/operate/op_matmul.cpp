#include "include/operate/operations.hpp"

#include "include/graph/func/functor.hpp"
#include "include/graph/func/coord_func.hpp"

#ifdef TENNCOR_OP_MATMUL_HPP

namespace nnet
{

static tensorshape matmul_shaper (std::vector<tensorshape> shapes)
{
	tensorshape& t1s = shapes[0];
	tensorshape& t2s = shapes[1];

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	// account for vectors
	size_t ax = rank1 ? al[0] : 0;
	size_t ay = rank1> 1 ? al[1] : 1;
	size_t bx = rank2 ? bl[0] : 0;
	size_t by = rank2> 1 ? bl[1] : 1;

	// ensure the dimensions beyond 2d are equal
	size_t minend = std::min(rank1, rank2);
	std::vector<size_t> beyond2d;
	if (minend> 2)
	{
		auto ait = al.begin()+2;
		auto aet = al.begin()+minend;
		if (std::equal(ait, aet, bl.begin()+2))
		{
			beyond2d.insert(beyond2d.end(), ait, aet);
		}
		else
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
		// check that remaining shape values are ones,
		// otherwise one shape is larger than the other
		auto it = rank1> rank2 ? al.begin() : bl.begin();
		auto et = rank1> rank2 ? al.end() : bl.end();
		if (!std::all_of(it + minend, et, [](size_t e) { return e == 1; }))
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple different sized shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
	}

	// get resulting shape
	std::vector<size_t> res_shape;
	if (ax == by)
	{
		res_shape = {bx, ay};
	}
	else
	{
		std::stringstream ss;
		ss << "matmul shapes ";
		print_shape(t1s, ss);
		ss << "and ";
		print_shape(t2s, ss);
		ss << "do not match";
		throw std::logic_error(ss.str());
	}
	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return res_shape;
}

static varptr flatten_mat (varptr a)
{
	size_t nelem_a = a->get_tensor()->get_shape().n_elems();
	varptr expand = coord_func({a},
	[](tensorshape, const tensorshape inshape, std::vector<uint64_t>)
	{
		size_t nelems = inshape.n_elems();
		std::vector<size_t> row(nelems);
		std::iota(row.begin(), row.end(), 0);
		std::vector<size_t> indices=row;
		for (size_t i = 1; i < nelems; ++i)
		{
			indices.insert(indices.end(), row.begin(), row.end());
		}
		return indices;
	},
	[](tensorshape inshape, std::vector<uint64_t>) -> tensorshape
	{
		size_t nelems = inshape.n_elems();
		return std::vector<size_t>{nelems, nelems};
	}, INJACOBIAN);
	std::vector<double> data(nelem_a * nelem_a, 0);
	for (size_t i = 0; i < nelem_a; ++i)
	{
		data[i * nelem_a + i] = 1;
	}
	varptr identity = constant::get<double>(data, std::vector<size_t>{nelem_a, nelem_a});
	return expand * identity;
}

static varptr matmul_gradient (inode* x, std::vector<inode*> args)
{
	inode* a = args.front();
	inode* b = args.back();
	inode* adx = a->derive(x);
	inode* bdx = b->derive(x);
	inode* c = matmul(varptr(a), varptr(b));
	// process both arguments as jacobians
	varptr da, db;
	tensorshape bases = x->get_tensor()->get_shape();
	if (adx)
	{
		adx = flatten_mat(adx); // todo: ensure adx keeps jacobian shape <xshape, cshape>
		// todo: check for parent ja
		varptr ja = functor::get({b, a, c},
		[](std::unique_ptr<idata_src>& src, std::vector<inode*> args)
		{
			inode* b = args[0];
			inode* a = args[1];
			inode* c = args[2];
			coord_io* osrc = new coord_io(
			[](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t>)
			{
				size_t xlimit = inshape[0];
				size_t ylimit = inshape[1];
				std::vector<signed> index(outshape.n_elems(), -1);
				size_t ns = inshape.n_elems();
				size_t nparts = outshape[0] / xlimit;
				for (size_t i = 0; i < ns; ++i)
				{
					std::vector<size_t> coord = inshape.coord_from_idx(i);
					size_t x = coord[0];
					size_t y = coord[1];
					for (size_t j = 0; j < nparts; ++j)
					{
						coord[0] = x + j * xlimit;
						coord[1] = y + j * ylimit;
						index[outshape.flat_idx(coord)] = i;
					}
				}
				return index;
			});
			src = std::unique_ptr<idata_src>(osrc);
			size_t nouter = a->get_tensor()->get_shape().n_elems();
			size_t ninner = c->get_tensor()->get_shape().n_elems();
			tensorshape outshape({ninner, nouter});
			b->get_tensor()->write_to(*osrc);
			return new tensor(outshape);
		},
		[](inode*, std::vector<inode*>) -> varptr
		{
			throw std::bad_function_call(); // unimplemented
		}, JACOBIANLEFT); // ja has shape <ashape, cshape>
		da = matmul(adx, ja); // da should have shape <xshape, ashape>
	}
	if (bdx)
	{
		bdx = flatten_mat(bdx);
		varptr jb = functor::get({a, b, c},
		[](std::unique_ptr<idata_src>& src, std::vector<inode*> args)
		{
			inode* a = args[0];
			inode* b = args[1];
			inode* c = args[2];
			coord_io* osrc = new coord_io(
			[](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t> blist)
			{
				size_t ylimit = inshape[0];
				size_t xlimit = blist[0];
				std::vector<signed> index(outshape.n_elems(), -1);
				size_t ns = inshape.n_elems();
				for (size_t i = 0; i < ns; ++i)
				{
					std::vector<size_t> coord = inshape.coord_from_idx(i);
					size_t x = coord[1] * ylimit;
					size_t y = coord[0] * xlimit;
					for (size_t j = 0; j < xlimit; ++j)
					{
						coord[0] = x + j;
						coord[1] = y + j;
						index[outshape.flat_idx(coord)] = i;
					}
				}
				return index;
			});
			src = std::unique_ptr<idata_src>(osrc);
			tensorshape bshape = b->get_tensor()->get_shape();
			size_t nouter = bshape.n_elems();
			size_t ninner = c->get_tensor()->get_shape().n_elems();
			tensorshape outshape({ninner, nouter});
			a->get_tensor()->write_to(*osrc);
			std::vector<size_t> blist = bshape.as_list();
			osrc->shape_info(std::vector<uint64_t>(blist.begin(), blist.end()));
			return new tensor(outshape);
		},
		[](inode*, std::vector<inode*>) -> varptr
		{
			throw std::bad_function_call(); // unimplemented
		}, JACOBIANRIGHT);
		db = matmul(bdx, jb);
	}
	// todo: ensure non-terminating matmul keep jacobian shape
	return coord_func({reduce_sum(da + db, 0)},
	[](tensorshape outshape, const tensorshape inshape, std::vector<uint64_t>)
	{
		std::vector<size_t> indices(inshape.n_elems());
		std::iota(indices.begin(), indices.end(), 0);
		return indices;
	},
	[bases](tensorshape, std::vector<uint64_t>)
	{
		return bases;
	}, OUTJACOBIAN);
}

varptr matmul (varptr a, varptr b)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;
	return functor::get({a, b}, 
	[](std::unique_ptr<idata_src>& src, std::vector<inode*> args)
	{
		idata_io* io = new operate_io("matmul");
		src = std::unique_ptr<idata_src>(io);
		const tensor* a = args.front()->get_tensor();
		const tensor* b = args.back()->get_tensor();
		a->write_to(*io, 0);
		b->write_to(*io, 1);
		tensorshape dshape = matmul_shaper({a->get_shape(), b->get_shape()});
		return new tensor(dshape);
	}, matmul_gradient, MATMUL);
}

}

#endif
