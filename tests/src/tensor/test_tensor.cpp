//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "tests/include/utils/fuzz.h"
#include "tests/include/mocks/mock_tensor.h"


#ifndef DISABLE_TENSOR_TEST


class TENSOR : public FUZZ::fuzz_test {};


static tensorshape random_partialshape (FUZZ::fuzz_test* fuzzer)
{
	std::vector<size_t> rlist = random_def_shape(fuzzer).as_list();
	size_t nzeros = fuzzer->get_int(1, "nzeros", {1, 5})[0];
	for (size_t i = 0; i < nzeros; i++)
	{
		size_t zidx = fuzzer->get_int(1, "zidx", {0, rlist.size()})[0];
		rlist.insert(rlist.begin()+zidx, 0);
	}
	return tensorshape(rlist);
}


// cover tensor: scalar constructor
TEST_F(TENSOR, ScalarConstructor_B000)
{
	std::vector<double> vals = get_double(3, "vals");
	double value = vals[0];
	mock_tensor scalar(value);
	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(scalar.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar.total_bytes());
	EXPECT_EQ(value, *scalar.rawptr());

	value = vals[1];
	mock_tensor scalar2(value);
	EXPECT_TRUE(scalar2.clean());
	EXPECT_TRUE(scalar2.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar2.total_bytes());
	EXPECT_EQ(value, *scalar2.rawptr());

	value = vals[2];
	mock_tensor scalar3(value);
	EXPECT_TRUE(scalar3.clean());
	EXPECT_TRUE(scalar3.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar3.total_bytes());
	EXPECT_EQ(value, *scalar3.rawptr());
}


// cover tensor:
// default, shape constructors,
// is_alloc, total_bytes
TEST_F(TENSOR, Construct_B001)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_tensor undef;
	mock_tensor incom(this, pshape);
	mock_tensor comp(this, cshape);

	EXPECT_TRUE(undef.clean());
	EXPECT_TRUE(incom.clean());
	EXPECT_TRUE(comp.clean());

	EXPECT_FALSE(undef.is_alloc());
	EXPECT_FALSE(incom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());

	EXPECT_EQ((size_t) 0, undef.total_bytes());
	EXPECT_EQ((size_t) 0, incom.total_bytes());
	EXPECT_EQ((size_t) sizeof(double) * cshape.n_elems(),
		comp.total_bytes());
}


// cover tensor:
// clone and assignment
TEST_F(TENSOR, Copy_B002)
{
	mock_tensor undefassign;
	mock_tensor scalarassign;
	mock_tensor incomassign;
	mock_tensor compassign;

	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_tensor undef;
	mock_tensor scalar(get_double(1, "scalar.data")[0]);
	mock_tensor incom(this, pshape);
	mock_tensor comp(this, cshape);

	mock_tensor undefcpy(undef);
	mock_tensor scalarcpy(scalar);
	mock_tensor incomcpy(incom);
	mock_tensor compcpy(comp);
	undefassign = undef;
	scalarassign = scalar;
	incomassign = incom;
	compassign = comp;

	EXPECT_FALSE(undefcpy.is_alloc());
	EXPECT_TRUE(scalarcpy.is_alloc());
	EXPECT_FALSE(incomcpy.is_alloc());
	EXPECT_TRUE(compcpy.is_alloc());
	EXPECT_FALSE(undefassign.is_alloc());
	EXPECT_TRUE(scalarassign.is_alloc());
	EXPECT_FALSE(incomassign.is_alloc());
	EXPECT_TRUE(compassign.is_alloc());

	EXPECT_TRUE(undefcpy.equal(undef));
	EXPECT_TRUE(scalarcpy.equal(scalar));
	EXPECT_TRUE(incomcpy.equal(incom));
	EXPECT_TRUE(compcpy.equal(comp));
	EXPECT_TRUE(undefassign.equal(undef));
	EXPECT_TRUE(scalarassign.equal(scalar));
	EXPECT_TRUE(incomassign.equal(incom));
	EXPECT_TRUE(compassign.equal(comp));
}


// cover tensor:
// move constructor and assignment
TEST_F(TENSOR, Move_B002)
{
	mock_tensor scalarassign;
	mock_tensor compassign;

	tensorshape sshape(std::vector<size_t>{1});
	tensorshape cshape = random_def_shape(this);
	mock_tensor scalar(get_double(1, "scalar.data")[0]);
	mock_tensor comp(this, cshape);

	const double* scalarptr = scalar.rawptr();
	const double* compptr = comp.rawptr();

	mock_tensor scalarmv(std::move(scalar));
	mock_tensor compmv(std::move(comp));

	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(comp.clean());
	EXPECT_TRUE(scalarmv.clean());
	EXPECT_TRUE(compmv.clean());

	EXPECT_FALSE(scalar.is_alloc());
	EXPECT_FALSE(comp.is_alloc());
	EXPECT_EQ(scalarptr, scalarmv.rawptr());
	EXPECT_EQ(compptr, compmv.rawptr());
	EXPECT_TRUE(scalarmv.allocshape_is(sshape));
	EXPECT_TRUE(compmv.allocshape_is(cshape));

	scalarassign = std::move(scalarmv);
	compassign = std::move(compmv);

	EXPECT_TRUE(scalarmv.clean());
	EXPECT_TRUE(compmv.clean());
	EXPECT_TRUE(scalarassign.clean());
	EXPECT_TRUE(compassign.clean());

	EXPECT_FALSE(scalarmv.is_alloc());
	EXPECT_FALSE(compmv.is_alloc());
	EXPECT_EQ(scalarptr, scalarassign.rawptr());
	EXPECT_EQ(compptr, compassign.rawptr());
	EXPECT_TRUE(scalarassign.allocshape_is(sshape));
	EXPECT_TRUE(compassign.allocshape_is(cshape));
}


// cover tensor:
// get_shape, n_elems
TEST_F(TENSOR, Shape_B003)
{
	tensorshape singular(std::vector<size_t>{1});
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_tensor undef;
	mock_tensor scalar(get_double(1, "scalar.data")[0]);
	mock_tensor incom(this, pshape);
	mock_tensor comp(this, cshape);

	EXPECT_TRUE(tensorshape_equal(undef.get_shape(), {}));
	EXPECT_TRUE(tensorshape_equal(singular, scalar.get_shape()));
	EXPECT_TRUE(tensorshape_equal(pshape, incom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, comp.get_shape()));

	EXPECT_EQ((size_t) 0, undef.n_elems());
	EXPECT_EQ((size_t) 1, scalar.n_elems());
	EXPECT_EQ((size_t) 0, incom.n_elems());
	EXPECT_EQ(cshape.n_elems(), comp.n_elems());
}


// cover tensor:
// get, expose
TEST_F(TENSOR, Get_B004)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);
	size_t crank = cshape.rank();
	size_t celem = cshape.n_elems();

	mock_tensor undef;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape);

	std::vector<double> cv = comp.expose(); // shouldn't die or throw
	// EXPECT_DEATH(undef.expose(), ".*");
	// EXPECT_DEATH(pcom.expose(), ".*");

	size_t pncoord = 1;
	if (crank > 2)
	{
		pncoord = get_int(1, "pncoord if crank > 3", {crank/2, crank-1})[0];
	}
	size_t cncoord = crank;
	size_t rncoord = get_int(1, "rncoord", {15, 127})[0];
	// c coordinates have rank exactly fitting cshape
	// p coordinates have rank less than rank of cshape
	// r coordinates are random coordinates
	std::vector<size_t> ccoord = get_int(cncoord, "ccoord");
	std::vector<size_t> pcoord = get_int(pncoord, "pcoord");
	std::vector<size_t> rcoord = get_int(rncoord, "rcoord");
	EXPECT_THROW(undef.get(pcoord), std::out_of_range);
	EXPECT_THROW(pcom.get(pcoord), std::out_of_range);
	EXPECT_THROW(undef.get(ccoord), std::out_of_range);
	EXPECT_THROW(pcom.get(ccoord), std::out_of_range);
	EXPECT_THROW(undef.get(rcoord), std::out_of_range);
	EXPECT_THROW(pcom.get(rcoord), std::out_of_range);

	std::vector<size_t> cs = cshape.as_list();
	size_t pcoordmax = 0, ccoordmax = 0, rcoordmax = 0;
	for (size_t i = 0, multiplier = 1, cn = cs.size(); i < cn; i++)
	{
		if (i < pncoord)
		{
			pcoordmax += pcoord[i] * multiplier;
		}
		if (i < rncoord)
		{
			rcoordmax += rcoord[i] * multiplier;
		}
		ccoordmax += ccoord[i] * multiplier;
		multiplier *= cs[i];
	}
	
	ASSERT_GT(celem, (size_t) 0);
	if (celem <= pcoordmax)
	{
		EXPECT_THROW(comp.get(pcoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), pcoordmax);
		EXPECT_EQ(cv[pcoordmax], comp.get(pcoord));
	}
	if (celem <= ccoordmax)
	{
		EXPECT_THROW(comp.get(ccoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), ccoordmax);
		EXPECT_EQ(cv[ccoordmax], comp.get(ccoord));
	}
	if (celem <= rcoordmax)
	{
		EXPECT_THROW(comp.get(rcoord), std::out_of_range);
	}
	else
	{
		ASSERT_GT(cv.size(), rcoordmax);
		EXPECT_EQ(cv[rcoordmax], comp.get(rcoord));
	}
}


// cover tensor: set_allocator
TEST_F(TENSOR, SetAlloc_B005)
{

}


// cover tensor: set_shape
TEST_F(TENSOR, Reshape_B006)
{
	tensorshape pshape = random_partialshape(this);
	// make cshape a 2d shape to make testing easy
	// todo: improve to test higher dimensionality
	tensorshape cshape = get_int(2, "cshape", {11, 127});
	std::vector<size_t> cv = cshape.as_list();
	size_t cols = cv[0];
	size_t rows = cv[1];

	mock_tensor undef;
	mock_tensor undef2;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape);
	mock_tensor comp2(this, cshape);
	mock_tensor comp3(this, cshape);
	mock_tensor comp4(this, cshape);

	// undefined/part defined shape change
	undef.set_shape(pshape);
	EXPECT_TRUE(tensorshape_equal(undef.get_shape(), pshape));
	EXPECT_FALSE(undef.is_alloc());

	undef2.set_shape(cshape);
	pcom.set_shape(cshape);
	EXPECT_TRUE(tensorshape_equal(undef2.get_shape(), cshape));
	EXPECT_TRUE(tensorshape_equal(pcom.get_shape(), cshape));
	EXPECT_FALSE(undef2.is_alloc());
	EXPECT_FALSE(pcom.is_alloc());

	ASSERT_TRUE(comp.is_alloc());
	ASSERT_TRUE(comp2.is_alloc());
	ASSERT_TRUE(comp3.is_alloc());
	ASSERT_TRUE(comp4.is_alloc());
	// comps must be alloc otherwise doubleDArr will fail assertion it != et
	// meaning data size < cv.n_elems
	std::vector<std::vector<double> > ac1 = doubleDArr(comp.expose(), cv);
	std::vector<std::vector<double> > ac2 = doubleDArr(comp2.expose(), cv);
	std::vector<std::vector<double> > ac3 = doubleDArr(comp3.expose(), cv);
	std::vector<std::vector<double> > ac4 = doubleDArr(comp4.expose(), cv);
	// data expansion
	std::vector<size_t> cvexp = cv;
	cvexp[0]++;
	cvexp[1]++;
	comp.set_shape(cvexp);
	std::vector<std::vector<double> > resc1 = doubleDArr(comp.expose(), cvexp);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			EXPECT_EQ(ac1[i][j], resc1[i][j]);
		}
		// check the padding
		EXPECT_EQ((size_t) 0, resc1[i][cols]);
	}
	// check the padding
	for (size_t i = 0; i < cols+1; i++)
	{
		EXPECT_EQ((size_t) 0, resc1[rows][i]);
	}

	// data clipping
	std::vector<size_t> cvcli = cv;
	cvcli[0]--;
	cvcli[1]--;
	comp2.set_shape(cvcli);
	std::vector<std::vector<double> > resc2 = doubleDArr(comp2.expose(), cvcli);
	for (size_t i = 0; i < rows-1; i++)
	{
		for (size_t j = 0; j < cols-1; j++)
		{
			EXPECT_EQ(ac2[i][j], resc2[i][j]);
		}
	}

	// clip in one dimension, expand in another
	std::vector<size_t> cvexpcli = cv;
	std::vector<size_t> cvexpcli2 = cv;
	cvexpcli[0]++;
	cvexpcli[1]--;
	cvexpcli2[0]--;
	cvexpcli2[1]++;
	comp3.set_shape(cvexpcli);
	comp4.set_shape(cvexpcli2);
	std::vector<std::vector<double> > resc3 = doubleDArr(comp3.expose(), cvexpcli);
	std::vector<std::vector<double> > resc4 = doubleDArr(comp4.expose(), cvexpcli2);
	for (size_t i = 0; i < rows-1; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			EXPECT_EQ(ac3[i][j], resc3[i][j]);
		}
		// check the padding
		EXPECT_EQ((size_t) 0, resc3[i][cols]);
	}
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols-1; j++)
		{
			EXPECT_EQ(ac4[i][j], resc4[i][j]);
		}
	}
	// check the padding
	for (size_t i = 0; i < cols-1; i++)
	{
		EXPECT_EQ((size_t) 0, resc4[rows][i]);
	}

	const double* p = comp.rawptr();
	const double* p2 = comp2.rawptr();
	tensorshape incmpshape = make_incompatible(comp.get_shape().as_list());
	comp.set_shape(incmpshape);
	comp2.set_shape(std::vector<size_t>{});
	ASSERT_NE(p, comp.rawptr());
	ASSERT_EQ(p2, comp2.rawptr());
}


// cover tensor:
// default allocate => bool allocate (void)
TEST_F(TENSOR, Allocate_B007)
{
	tensorshape cshape = random_def_shape(this);
	tensorshape pshape = random_partialshape(this);

	mock_tensor undef;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape);
	ASSERT_TRUE(comp.is_alloc());
	ASSERT_FALSE(pcom.is_alloc());
	ASSERT_FALSE(undef.is_alloc());
	const double* orig = comp.rawptr(); // check to see if comp.rawptr changes later
	EXPECT_FALSE(undef.allocate());
	EXPECT_FALSE(pcom.allocate());
	EXPECT_FALSE(comp.allocate());
	// change allowed shape to defined shape, cshape
	undef.set_shape(cshape);
	pcom.set_shape(cshape);
	EXPECT_TRUE(undef.allocate());
	EXPECT_TRUE(pcom.allocate());
	EXPECT_EQ(orig, comp.rawptr());
}


// cover tensor: deallocate
TEST_F(TENSOR, Dealloc_B008)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_tensor undef;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape);

	EXPECT_FALSE(undef.is_alloc());
	EXPECT_FALSE(pcom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());
	EXPECT_FALSE(undef.deallocate());
	EXPECT_FALSE(pcom.deallocate());
	EXPECT_TRUE(comp.deallocate());
	EXPECT_FALSE(comp.is_alloc());
}


// cover tensor:
// allocate shape => bool allocate (const tensorshape shape)
TEST_F(TENSOR, AllocateShape_B009)
{
	tensorshape cshape = random_def_shape(this);
	std::vector<size_t> cv = cshape.as_list();
	tensorshape cshape2 = make_incompatible(cv);
	tensorshape pshape = make_partial(this, cv);
	tensorshape pshape2 = make_full_incomp(pshape.as_list(), cv);

	mock_tensor undef;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape);
	const double* orig = comp.rawptr();

	EXPECT_FALSE(undef.allocate(pshape));
	EXPECT_TRUE(undef.allocate(cshape));
	EXPECT_FALSE(pcom.allocate(cshape2));
	EXPECT_TRUE(pcom.allocate(cshape));
	EXPECT_FALSE(comp.allocate(cshape));
	EXPECT_FALSE(comp.allocate(cshape2));
	EXPECT_FALSE(comp.allocate(pshape));
	EXPECT_EQ(orig, comp.rawptr());

	EXPECT_TRUE(tensorshape_equal(cshape, undef.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, pcom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape, comp.get_shape()));

	// they're all allocated now
	EXPECT_TRUE(undef.is_alloc());
	EXPECT_TRUE(pcom.is_alloc());
	EXPECT_TRUE(comp.is_alloc());

	ASSERT_TRUE(pcom.allocate(pshape2));
	EXPECT_NE(orig, pcom.rawptr());
}


// cover tensor: copy_from
TEST_F(TENSOR, CopyWithShape_B010)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);
	tensorshape cshape2 = random_def_shape(this);
	tensorshape cshape3 = random_def_shape(this);

	size_t n1 = cshape.n_elems();
	std::vector<double> rawdata1 = get_double(n1, "rawdata1");
	size_t n2 = cshape2.n_elems();
	std::vector<double> rawdata2 = get_double(n2, "rawdata2");
	mock_tensor undef;
	mock_tensor pcom(this, pshape);
	mock_tensor comp(this, cshape, rawdata1);
	mock_tensor comp2(this, cshape2, rawdata2);
	const double* orig = comp.rawptr();
	const double* orig2 = comp2.rawptr();
	std::vector<double> compdata = comp.expose();
	std::vector<double> compdata2 = comp2.expose();

	// copying from unallocated
	EXPECT_FALSE(pcom.copy_from(undef, cshape));
	EXPECT_FALSE(undef.copy_from(pcom, cshape));
	EXPECT_FALSE(pcom.is_alloc());
	EXPECT_FALSE(undef.is_alloc());

	EXPECT_TRUE(undef.copy_from(comp, cshape3));
	EXPECT_TRUE(pcom.copy_from(comp2, cshape3));

	EXPECT_TRUE(comp.copy_from(comp2, cshape3));
	EXPECT_TRUE(comp2.copy_from(comp2, cshape3)); // copy from self

	// pointers are now different
	EXPECT_NE(orig, comp.rawptr());
	EXPECT_NE(orig2, comp2.rawptr());

	EXPECT_TRUE(tensorshape_equal(cshape3, undef.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, pcom.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, comp.get_shape()));
	EXPECT_TRUE(tensorshape_equal(cshape3, comp2.get_shape()));

	std::vector<double> undefdata = undef.expose();
	std::vector<double> pdefdata = pcom.expose();

	std::vector<size_t> c1list = cshape.as_list();
	std::vector<size_t> c2list = cshape2.as_list();
	std::vector<size_t> c3list = cshape3.as_list();

	// undef fitted with comp and cshape3
	for (size_t i = 0, n = cshape.n_elems(); i < n; i++)
	{
		std::vector<size_t> incoord = cshape.coordinate_from_idx(i);
		bool b = true;
		for (size_t j = 0, o = incoord.size(); j < o && b; j++)
		{
			if (j >= c3list.size())
			{
				b = incoord[j] == 0;
			}
			else
			{
				b = incoord[j] < c3list[j];
			}
		}
		if (b)
		{
			size_t outidx = cshape3.flat_idx(incoord);
			EXPECT_EQ(compdata[i], undefdata[outidx]);
		}
	}
	// pdefdata fitted with comp2 and cshape 3
	for (size_t i = 0, n = cshape2.n_elems(); i < n; i++)
	{
		std::vector<size_t> incoord = cshape2.coordinate_from_idx(i);
		bool b = true;
		for (size_t j = 0, o = incoord.size(); j < o && b; j++)
		{
			if (j >= c3list.size())
			{
				b = incoord[j] == 0;
			}
			else
			{
				b = incoord[j] < c3list[j];
			}
		}
		if (b)
		{
			size_t outidx = cshape3.flat_idx(incoord);
			EXPECT_EQ(compdata2[i], pdefdata[outidx]);
		}
	}
}


#endif /* DISABLE_TENSOR_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */
