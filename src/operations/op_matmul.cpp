#include "include/operations/operations.hpp"

#include "include/graph/connector/functor.hpp"
#include "include/graph/connector/immutable/coord_mapper.hpp"

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
	varptr expand = coord_mapper::get(a,
	[](tensorshape, const tensorshape inshape)
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
	[](tensorshape inshape) -> tensorshape
	{
		size_t nelems = inshape.n_elems();
		return std::vector<size_t>{nelems, nelems};
	}, "expand");
	std::vector<double> data(nelem_a * nelem_a, 0);
	for (size_t i = 0; i < nelem_a; ++i)
	{
		data[i * nelem_a + i] = 1;
	}
	varptr identity = constant::get(data, std::vector<size_t>{nelem_a, nelem_a});
	return expand * identity;
}

static inline varptr select (tensorshape shape, size_t index)
{
	std::vector<double> data(shape.n_elems(), 0);
	data[index] = 1;
	return constant::get(data, shape);
}

static inline void jacobian_glue (VARR_T dest, CVAR_T src, unsigned short nbyte, size_t index)
{
	char* cdest = (char*) dest.first;
	const char* csrc = (const char*) src.first;
	size_t ncol = dest.second.as_list()[0];
	size_t nrow = src.second.n_elems();
	for (size_t i = 0; i < nrow; ++i)
	{
		std::memcpy(cdest + (i * ncol + index) * nbyte, csrc + i * nbyte, nbyte);
	}
}

static varptr matmul_gradient (inode* x, std::vector<inode*> args, inode* c)
{
    inode* a = args.front();
    inode* b = args.back();
    inode* adx = a->derive(x);
    inode* bdx = b->derive(x);
    // process both arguments as jacobians
    varptr da, db;
    tensorshape bases = x->get_tensor()->get_shape();
    if (adx)
    {
        adx = flatten_mat(adx);
        // todo: check for parent ja
        varptr ja = functor::get({a, b, c},
        [](std::unique_ptr<idata_src>& src, std::vector<inode*> args)
        {
            inode* a = args[0];
            inode* b = args[1];
            inode* c = args[2];
            idata_io* io = new glue_io(jacobian_glue);
            src = std::unique_ptr<idata_src>(io);
            tensorshape outershape = a->get_tensor()->get_shape();
            size_t nouter = outershape.n_elems();
            size_t ninner = c->get_tensor()->get_shape().n_elems();
            for (size_t i = 0; i < nouter; ++i)
            {
                varptr ivar = matmul(select(outershape, i), varptr(b));
                tensor* iten = ivar->get_tensor();
                iten->write_to(*io, i);
            }
            return new tensor(tensorshape({nouter, ninner}));
        }, 
        [](inode*, std::vector<inode*>, inode*) -> varptr
        {
            throw std::bad_function_call(); // unimplemented
        }, "jacobian_a");
        da = matmul(ja, adx);
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
            idata_io* io = new glue_io(jacobian_glue);
            src = std::unique_ptr<idata_src>(io);
            tensorshape outershape = b->get_tensor()->get_shape();
            size_t nouter = outershape.n_elems();
            size_t ninner = c->get_tensor()->get_shape().n_elems();
            for (size_t i = 0; i < nouter; ++i)
            {
                varptr ivar = matmul(varptr(a), select(outershape, i));
                tensor* iten = ivar->get_tensor();
                iten->write_to(*io, i);
            }
            return new tensor(tensorshape({nouter, ninner}));
        },
        [](inode*, std::vector<inode*>, inode*) -> varptr
        {
            throw std::bad_function_call(); // unimplemented
        }, "jacobian_b");
        db = matmul(jb, bdx);
    }
    return coord_mapper::get(reduce_sum(da + db, 1),
    [](tensorshape outshape, const tensorshape inshape)
    {
        std::vector<size_t> indices(inshape.n_elems());
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    },
    [bases](tensorshape)
    {
        return bases;
    }, "jacobian_expand");
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
	}, matmul_gradient, "matmul");
}

}

#endif
