#ifndef DISABLE_SLIP_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "slip/include/shaper.hpp"
#include "slip/error.hpp"

#include "clay/memory.hpp"


#ifndef DISABLE_SHAPER_TEST


using namespace testutil;


class SHAPER : public fuzz_test {};


// put inlist between outlist splitting outlist at index i
std::vector<size_t> inject (std::vector<size_t> outlist, std::vector<size_t> inlist, size_t i)
{
	auto it = outlist.begin();
	std::vector<size_t> out(it, it + i);
	out.insert(out.end(), inlist.begin(), inlist.end());
	out.insert(out.end(), it + i, outlist.end());
	return out;
}


TEST_F(SHAPER, BoringElem_A000)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	size_t n = shape.n_elems();
	size_t nbytes = n * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::shared_ptr<char> data2 = clay::make_char(nbytes);

	clay::Shape wshape({1});
	std::shared_ptr<char> wun = clay::make_char(sizeof(double));
	*((double*) wun.get()) = get_double(1, "scalar")[0];

	size_t sidx = get_int(1, "sidx")[0];
	mold::Range srange(sidx, sidx);
	// boring scalar ranged
	clay::Shape ushape = slip::elem_shape({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), srange)
	});

	clay::Shape binshape = slip::elem_shape({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(data2, shape, clay::DOUBLE), srange)
	});

	EXPECT_SHAPEQ(shape, ushape);
	EXPECT_SHAPEQ(shape, binshape);

	// boring scalar vs scalar ranged
	binshape = slip::elem_shape({
		mold::StateRange(clay::State(wun, wshape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(data2, shape, clay::DOUBLE), srange)
	});
	EXPECT_SHAPEQ(shape, binshape);
	binshape = slip::elem_shape({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(wun, shape, clay::DOUBLE), srange)
	});
	EXPECT_SHAPEQ(shape, binshape);

	// boring ranged vs scalar ranged
	std::vector<size_t> alist = random_def_shape(this);
	std::vector<size_t> blist = random_def_shape(this);
	std::vector<size_t> indices = get_int(2, "indices", {0, clist.size()});
	mold::Range arange(indices[0], indices[0] + alist.size());
	mold::Range brange(indices[1], indices[1] + blist.size());

	clay::Shape ashape = inject(clist, alist, indices[0]);
	clay::Shape bshape = inject(clist, blist, indices[1]);
	std::shared_ptr<char> adata = clay::make_char(ashape.n_elems() * sizeof(double));
	std::shared_ptr<char> bdata2 = clay::make_char(bshape.n_elems() * sizeof(double));

	clay::Shape srashape = slip::elem_shape({
		mold::StateRange(clay::State(adata, ashape, clay::DOUBLE), arange),
		mold::StateRange(clay::State(data2, shape, clay::DOUBLE), srange)
	});
	clay::Shape srbshape = slip::elem_shape({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(bdata2, bshape, clay::DOUBLE), brange)
	});

	EXPECT_SHAPEQ(ashape, srashape);
	EXPECT_SHAPEQ(bshape, srbshape);

	// boring ranged vs scalar ranged
	srashape = slip::elem_shape({
		mold::StateRange(clay::State(adata, ashape, clay::DOUBLE), arange),
		mold::StateRange(clay::State(wun, wshape, clay::DOUBLE), srange)
	});
	EXPECT_SHAPEQ(ashape, srashape);
	srbshape = slip::elem_shape({
		mold::StateRange(clay::State(wun, wshape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(bdata2, bshape, clay::DOUBLE), brange)
	});
	EXPECT_SHAPEQ(bshape, srbshape);

	// boring common ranged
	std::vector<size_t> ilist = random_def_shape(this);
	size_t index = get_int(1, "index", {0, clist.size()})[0];
	mold::Range rrange(index, index + ilist.size());
	clay::Shape oshape = inject(clist, ilist, index);
	std::shared_ptr<char> odata = clay::make_char(oshape.n_elems() * sizeof(double));
	std::shared_ptr<char> odata2 = clay::make_char(oshape.n_elems() * sizeof(double));

	clay::Shape rshape = slip::elem_shape({
		mold::StateRange(clay::State(odata, oshape, clay::DOUBLE), rrange),
		mold::StateRange(clay::State(odata2, oshape, clay::DOUBLE), rrange)
	});
	EXPECT_SHAPEQ(oshape, rshape);
}


TEST_F(SHAPER, BadElem_A000)
{
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> blist = make_incompatible(clist);
	clay::Shape shape = clist;
	clay::Shape shape2 = blist;
	size_t n = shape.n_elems();
	size_t n2 = shape2.n_elems();
	size_t nbytes = n * sizeof(double);
	size_t nbytes2 = n2 * sizeof(double);
	std::shared_ptr<char> data = clay::make_char(nbytes);
	std::shared_ptr<char> data2 = clay::make_char(nbytes2);

	size_t sidx = get_int(1, "sidx")[0];
	mold::Range srange(sidx, sidx);
	// scalar ranged
	EXPECT_THROW(slip::elem_shape({
		mold::StateRange(clay::State(data, shape, clay::DOUBLE), srange),
		mold::StateRange(clay::State(data2, shape2, clay::DOUBLE), srange)
	}), slip::ShapeMismatchError);

	std::vector<size_t> inalist = random_def_shape(this);
	std::vector<size_t> inblist = make_incompatible(inalist);

	std::vector<size_t> indices = get_int(2, "indices", {0, inalist.size()});
	mold::Range arange(indices[0], indices[0] + inalist.size());
	mold::Range brange(indices[1], indices[1] + inblist.size());
	// incompatible outershape
	clay::Shape ashape = inject(clist, inalist, indices[0]);
	clay::Shape bshape = inject(blist, inalist, indices[1]);
	EXPECT_THROW(slip::elem_shape({
		mold::StateRange(clay::State(data, ashape, clay::DOUBLE), arange),
		mold::StateRange(clay::State(data2, bshape, clay::DOUBLE), brange)
	}), slip::ShapeMismatchError);

	// incompatible innershape
	clay::Shape ashape2 = inject(clist, inalist, indices[0]);
	clay::Shape bshape2 = inject(clist, inblist, indices[1]);
	EXPECT_THROW(slip::elem_shape({
		mold::StateRange(clay::State(data, ashape2, clay::DOUBLE), arange),
		mold::StateRange(clay::State(data2, bshape2, clay::DOUBLE), brange)
	}), slip::ShapeMismatchError);
}


#endif /* DISABLE_SHAPER_TEST */


#endif /* DISABLE_SLIP_MODULE_TESTS */
