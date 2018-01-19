//
// Created by Mingkai Chen on 2018-01-17.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include "gtest/gtest.h"

#include "tests/include/utils/fuzz.h"
#include "tests/include/mocks/mock_itensor.h"

#ifndef DISABLE_ITENSOR_TEST


class ITENSOR : public FUZZ::fuzz_test {};


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


// cover itensor: scalar constructor
TEST_F(ITENSOR, ScalarConstructor_C000)
{
	std::vector<double> vals = get_double(3, "vals");
	double value = vals[0];
	mock_itensor scalar(value);
	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(scalar.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar.total_bytes());
	EXPECT_EQ(value, *scalar.rawptr());

	value = vals[1];
	mock_itensor scalar2(value);
	EXPECT_TRUE(scalar2.clean());
	EXPECT_TRUE(scalar2.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar2.total_bytes());
	EXPECT_EQ(value, *scalar2.rawptr());

	value = vals[2];
	mock_itensor scalar3(value);
	EXPECT_TRUE(scalar3.clean());
	EXPECT_TRUE(scalar3.is_alloc());
	EXPECT_EQ((size_t) sizeof(double), scalar3.total_bytes());
	EXPECT_EQ(value, *scalar3.rawptr());
}


// cover itensor:
// default, shape constructors,
// is_alloc, total_bytes
TEST_F(ITENSOR, Construct_C001)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_itensor undef;
	mock_itensor incom(this, pshape);
	mock_itensor comp(this, cshape);

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


// cover itensor:
// clone and assignment
TEST_F(ITENSOR, Copy_C002)
{
	mock_itensor undefassign;
	mock_itensor scalarassign;
	mock_itensor incomassign;
	mock_itensor compassign;

	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_itensor undef;
	mock_itensor scalar(get_double(1, "scalar.data")[0]);
	mock_itensor incom(this, pshape);
	mock_itensor comp(this, cshape);

	mock_itensor* undefcpy = undef.clone();
	mock_itensor* scalarcpy = scalar.clone();
	mock_itensor* incomcpy = incom.clone();
	mock_itensor* compcpy = comp.clone();
	undefassign = undef;
	scalarassign = scalar;
	incomassign = incom;
	compassign = comp;

	EXPECT_FALSE(undefcpy->is_alloc());
	EXPECT_TRUE(scalarcpy->is_alloc());
	EXPECT_FALSE(incomcpy->is_alloc());
	EXPECT_TRUE(compcpy->is_alloc());
	EXPECT_FALSE(undefassign.is_alloc());
	EXPECT_TRUE(scalarassign.is_alloc());
	EXPECT_FALSE(incomassign.is_alloc());
	EXPECT_TRUE(compassign.is_alloc());

	EXPECT_TRUE(undefcpy->equal(undef));
	EXPECT_TRUE(scalarcpy->equal(scalar));
	EXPECT_TRUE(incomcpy->equal(incom));
	EXPECT_TRUE(compcpy->equal(comp));
	EXPECT_TRUE(undefassign.equal(undef));
	EXPECT_TRUE(scalarassign.equal(scalar));
	EXPECT_TRUE(incomassign.equal(incom));
	EXPECT_TRUE(compassign.equal(comp));

	delete undefcpy;
	delete scalarcpy;
	delete incomcpy;
	delete compcpy;
}


// cover itensor:
// move constructor and assignment
TEST_F(ITENSOR, Move_C002)
{
	mock_itensor scalarassign;
	mock_itensor compassign;

	tensorshape sshape(std::vector<size_t>{1});
	tensorshape cshape = random_def_shape(this);
	mock_itensor scalar(get_double(1, "scalar.data")[0]);
	mock_itensor comp(this, cshape);

	const double* scalarptr = scalar.rawptr();
	const double* compptr = comp.rawptr();

	mock_itensor* scalarmv = scalar.move();
	mock_itensor* compmv = comp.move();

	EXPECT_TRUE(scalar.clean());
	EXPECT_TRUE(comp.clean());
	EXPECT_TRUE(scalarmv->clean());
	EXPECT_TRUE(compmv->clean());

	EXPECT_FALSE(scalar.is_alloc());
	EXPECT_FALSE(comp.is_alloc());
	EXPECT_EQ(scalarptr, scalarmv->rawptr());
	EXPECT_EQ(compptr, compmv->rawptr());
	EXPECT_TRUE(scalarmv->allocshape_is(sshape));
	EXPECT_TRUE(compmv->allocshape_is(cshape));

	scalarassign = std::move(*scalarmv);
	compassign = std::move(*compmv);

	EXPECT_TRUE(scalarmv->clean());
	EXPECT_TRUE(compmv->clean());
	EXPECT_TRUE(scalarassign.clean());
	EXPECT_TRUE(compassign.clean());

	EXPECT_FALSE(scalarmv->is_alloc());
	EXPECT_FALSE(compmv->is_alloc());
	EXPECT_EQ(scalarptr, scalarassign.rawptr());
	EXPECT_EQ(compptr, compassign.rawptr());
	EXPECT_TRUE(scalarassign.allocshape_is(sshape));
	EXPECT_TRUE(compassign.allocshape_is(cshape));

	delete scalarmv;
	delete compmv;
}


// cover itensor:
// rank. dims
TEST_F(ITENSOR, Shape_C003)
{
	tensorshape singular(std::vector<size_t>{1});
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_itensor undef;
	mock_itensor scalar(get_double(1, "scalar.data")[0]);
	mock_itensor incom(this, pshape);
	mock_itensor comp(this, cshape);

	EXPECT_EQ((size_t) 0, undef.rank());
	EXPECT_EQ((size_t) 1, scalar.rank());
	EXPECT_EQ(pshape.rank(), incom.rank());
	EXPECT_EQ(cshape.rank(), comp.rank());

	EXPECT_TRUE(undef.dims().empty());
	std::vector<size_t> sv = scalar.dims();
	ASSERT_EQ((size_t) 1, sv.size());
	EXPECT_EQ((size_t) 1, sv[0]);

	std::vector<size_t> expects = pshape.as_list();
	std::vector<size_t> expectc = cshape.as_list();
	EXPECT_TRUE(std::equal(expects.begin(), expects.end(), incom.dims().begin()));
	EXPECT_TRUE(std::equal(expectc.begin(), expectc.end(), comp.dims().begin()));
}


// cover itensor: is_same_size
TEST_F(ITENSOR, IsSameSize_C004)
{
	tensorshape cshape = random_def_shape(this);
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	mock_itensor bad(this, ishape);
	mock_itensor undef;
	mock_itensor scalar(get_double(1, "scalar.data")[0]);
	mock_itensor comp(this, cshape);

	{
		tensorshape pshape = make_partial(this, cv); // same as cshape
		mock_itensor pcom(this, pshape);
		// allowed compatible
		// pcom, undef are both unallocated
		EXPECT_FALSE(undef.is_alloc());
		EXPECT_FALSE(pcom.is_alloc());
		// undef is same as anything
		EXPECT_TRUE(undef.is_same_size(bad));
		EXPECT_TRUE(undef.is_same_size(comp));
		EXPECT_TRUE(undef.is_same_size(scalar));
		EXPECT_TRUE(undef.is_same_size(pcom));
		// pcom is same as comp, but not bad or scalar
		EXPECT_TRUE(pcom.is_same_size(comp));
		EXPECT_FALSE(pcom.is_same_size(bad));
		EXPECT_FALSE(pcom.is_same_size(scalar));
	}

	// trimmed compatible
	{
		// padd cv
		std::vector<size_t> npads = get_int(4, "npads", {3, 17});
		tensorshape p1 = padd(cv, npads[0], npads[1]); // same
		tensorshape p2 = padd(cv, npads[2], npads[3]); // same
		cv.push_back(2);
		tensorshape p3 = padd(cv, npads[2], npads[3]); // not same
		mock_itensor comp2(this, p1);
		mock_itensor comp3(this, p2);
		mock_itensor bad2(this, p3);

		EXPECT_TRUE(comp2.is_alloc());
		EXPECT_TRUE(comp3.is_alloc());
		EXPECT_TRUE(bad.is_alloc());

		EXPECT_TRUE(comp.is_same_size(comp2));
		EXPECT_TRUE(comp2.is_same_size(comp3));
		EXPECT_TRUE(comp.is_same_size(comp3));

		EXPECT_FALSE(comp.is_same_size(bad));
		EXPECT_FALSE(comp2.is_same_size(bad));
		EXPECT_FALSE(comp3.is_same_size(bad));

		EXPECT_FALSE(comp.is_same_size(bad2));
		EXPECT_FALSE(comp2.is_same_size(bad2));
		EXPECT_FALSE(comp3.is_same_size(bad2));
	}
}


// cover itensor: 
// is_compatible_with tensor => bool is_compatible_with (const itensor& other) const
TEST_F(ITENSOR, IsCompatibleWithTensor_C005)
{
	tensorshape cshape = random_def_shape(this);
	std::vector<size_t> cv = cshape.as_list();
	tensorshape ishape = make_incompatible(cv); // not same as cshape
	tensorshape pshape = make_partial(this, cv); // same as cshape
	mock_itensor undef;
	mock_itensor scalar(get_double(1, "scalar.data")[0]);
	mock_itensor comp(this, cshape);
	mock_itensor pcom(this, pshape);
	mock_itensor bad(this, ishape);

	// undefined tensor is compatible with anything
	EXPECT_TRUE(undef.is_compatible_with(undef));
	EXPECT_TRUE(undef.is_compatible_with(scalar));
	EXPECT_TRUE(undef.is_compatible_with(comp));
	EXPECT_TRUE(undef.is_compatible_with(pcom));
	EXPECT_TRUE(undef.is_compatible_with(bad));

	EXPECT_TRUE(pcom.is_compatible_with(comp));
	EXPECT_TRUE(pcom.is_compatible_with(pcom));
	EXPECT_FALSE(pcom.is_compatible_with(bad));

	EXPECT_FALSE(bad.is_compatible_with(comp));
}


// cover itensor:
// is_compatible_with vector => bool is_compatible_with (size_t ndata) const,
// is_loosely_compatible_with
TEST_F(ITENSOR, IsCompatibleWithVector_C006)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);

	mock_itensor undef;
	mock_itensor comp(this, cshape);
	mock_itensor pcom(this, pshape);

	size_t exactdata = cshape.n_elems();
	size_t lowerdata = 1;
	if (exactdata >= 3)
	{
		lowerdata = exactdata - get_int(1, 
			"exactdata - lowerdata", {1, exactdata-1})[0];
	}
	size_t upperdata = exactdata + get_int(1, 
		"upperdata - exactdata", {1, exactdata-1})[0];

	EXPECT_TRUE(comp.is_compatible_with(exactdata));
	EXPECT_FALSE(comp.is_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_compatible_with(upperdata));

	EXPECT_TRUE(comp.is_loosely_compatible_with(exactdata));
	EXPECT_TRUE(comp.is_loosely_compatible_with(lowerdata));
	EXPECT_FALSE(comp.is_loosely_compatible_with(upperdata));

	size_t exactdata2 = pshape.n_known();
	size_t lowerdata2 = 1;
	if (exactdata2 >= 3)
	{
		lowerdata2 = exactdata2 - get_int(1, 
			"exactdata2 - lowerdata2", {1, exactdata2-1})[0];
	}
	size_t moddata = exactdata2 * get_int(1, 
		"moddata / exactdata2", {2, 15})[0];
	size_t upperdata2 = moddata + 1;

	EXPECT_TRUE(pcom.is_compatible_with(exactdata2));
	EXPECT_TRUE(pcom.is_compatible_with(moddata));
	EXPECT_FALSE(pcom.is_compatible_with(lowerdata2));
	EXPECT_FALSE(pcom.is_compatible_with(upperdata2));

	EXPECT_TRUE(pcom.is_loosely_compatible_with(exactdata2));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(moddata));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(lowerdata2));
	EXPECT_TRUE(pcom.is_loosely_compatible_with(upperdata2));

	// undef is compatible with everything
	EXPECT_TRUE(undef.is_compatible_with(exactdata));
	EXPECT_TRUE(undef.is_compatible_with(exactdata2));
	EXPECT_TRUE(undef.is_compatible_with(lowerdata));
	EXPECT_TRUE(undef.is_compatible_with(lowerdata2));
	EXPECT_TRUE(undef.is_compatible_with(upperdata));
	EXPECT_TRUE(undef.is_compatible_with(upperdata2));
	EXPECT_TRUE(undef.is_compatible_with(moddata));

	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata));
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata2));
	EXPECT_TRUE(undef.is_loosely_compatible_with(moddata));
}


// covers tensor
// guess_shape, loosely_guess_shape
TEST_F(ITENSOR, GuessShape_C007)
{
	tensorshape pshape = random_partialshape(this);
	tensorshape cshape = random_def_shape(this);
	mock_itensor undef;
	mock_itensor comp(this, cshape);
	mock_itensor pcom(this, pshape);

	size_t exactdata = cshape.n_elems();
	size_t lowerdata = 1;
	if (exactdata >= 3)
	{
		lowerdata = exactdata - get_int(1, 
			"exactdata - lowerdata", {1, exactdata-1})[0];
	}
	size_t upperdata = exactdata + get_int(1, 
		"upperdata - exactdata", {1, exactdata-1})[0];

	// allowed are fully defined
	optional<tensorshape> cres = comp.guess_shape(exactdata);
	ASSERT_TRUE((bool)cres);
	EXPECT_TRUE(tensorshape_equal(cshape, *cres));
	EXPECT_FALSE((bool)comp.guess_shape(lowerdata));
	EXPECT_FALSE((bool)comp.guess_shape(upperdata));

	size_t exactdata2 = pshape.n_known();
	size_t lowerdata2 = 1;
	if (exactdata2 >= 3)
	{
		lowerdata2 = exactdata2 - get_int(1, 
			"exactdata2 - lowerdata2", {1, exactdata2-1})[0];
	}
	size_t moddata = exactdata2 * get_int(1, "moddata / exactdata2", {2, 15})[0];
	size_t upperdata2 = moddata + 1;

	std::vector<size_t> pv = pshape.as_list();
	size_t unknown = pv.size();
	for (size_t i = 0; i < pv.size(); i++)
	{
		if (0 == pv[i])
		{
			if (unknown > i)
			{
				unknown = i;
			}
			pv[i] = 1;
		}
	}
	std::vector<size_t> pv2 = pv;
	pv2[unknown] = ceil((double) moddata / (double) exactdata2);
	// allowed are partially defined
	optional<tensorshape> pres = pcom.guess_shape(exactdata2);
	optional<tensorshape> pres2 = pcom.guess_shape(moddata);
	ASSERT_TRUE((bool)pres);
	ASSERT_TRUE((bool)pres2);
	EXPECT_TRUE(tensorshape_equal(*pres, pv));
	EXPECT_TRUE(tensorshape_equal(*pres2, pv2));
	EXPECT_FALSE((bool)pcom.guess_shape(lowerdata2));
	EXPECT_FALSE((bool)pcom.guess_shape(upperdata2));

	// allowed are undefined
	optional<tensorshape> ures = undef.guess_shape(exactdata);
	optional<tensorshape> ures2 = undef.guess_shape(exactdata2);
	optional<tensorshape> ures3 = undef.guess_shape(lowerdata);
	optional<tensorshape> ures4 = undef.guess_shape(lowerdata2);
	optional<tensorshape> ures5 = undef.guess_shape(upperdata);
	optional<tensorshape> ures6 = undef.guess_shape(upperdata2);
	optional<tensorshape> ures7 = undef.guess_shape(moddata);
	ASSERT_TRUE((bool)ures);
	ASSERT_TRUE((bool)ures2);
	ASSERT_TRUE((bool)ures3);
	ASSERT_TRUE((bool)ures4);
	ASSERT_TRUE((bool)ures5);
	ASSERT_TRUE((bool)ures6);
	ASSERT_TRUE((bool)ures7);
	EXPECT_TRUE(tensorshape_equal(*ures, std::vector<size_t>({exactdata})));
	EXPECT_TRUE(tensorshape_equal(*ures2, std::vector<size_t>({exactdata2})));
	EXPECT_TRUE(tensorshape_equal(*ures3, std::vector<size_t>({lowerdata})));
	EXPECT_TRUE(tensorshape_equal(*ures4, std::vector<size_t>({lowerdata2})));
	EXPECT_TRUE(tensorshape_equal(*ures5, std::vector<size_t>({upperdata})));
	EXPECT_TRUE(tensorshape_equal(*ures6, std::vector<size_t>({upperdata2})));
	EXPECT_TRUE(tensorshape_equal(*ures7, std::vector<size_t>({moddata})));
}


// covers itensor: 
// serialize,
// from_proto without alloc reassign => bool from_proto (const tenncor::tensor_proto& other)
TEST_F(ITENSOR, Serialize_C008)
{
	
}


#endif /* DISABLE_ITENSOR_TEST */

#endif /* DISABLE_TENSOR_MODULE_TESTS */
