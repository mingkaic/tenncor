//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include <algorithm>
#include <sstream>

#include "gtest/gtest.h"

#include "tests/utils/sgen.hpp"
#include "tests/utils/check.hpp"

#include "mocker/mocker.hpp"

#include "tensor/tensor.hpp"

#ifndef DISABLE_TENSOR_TEST


class TENSOR : public testify::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testify::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


using namespace testutils;


struct mock_data_src final : public nnet::idata_src, public testify::mocker
{
	mock_data_src (testify::fuzz_test* fuzzer) :
		type_((TENS_TYPE) fuzzer->get_int(1, "type", {1, N_TYPE - 1})[0]),
		uuid_(fuzzer->get_string(16, "mock_src_uuid")) {}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, nnet::tensorshape shape) const
	{
		outptr = nnutils::make_svoid(uuid_.size());
		std::memcpy(outptr.get(), &uuid_[0], uuid_.size());
		type = type_;

		label_incr("get_data");
		std::stringstream ss;
		print_shape(shape, ss);
		set_label("get_data", ss.str());
	}

	TENS_TYPE type_;

	std::string uuid_;

private:
	virtual idata_src* clone_impl (void) const
	{
		return nullptr;
	}
};


struct mock_data_dest final : public nnet::idata_dest, public testify::mocker
{
	mock_data_dest (mock_data_src& src) : expect_result_(src.uuid_), result_(16, ' ') {}

	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, nnet::tensorshape shape, size_t idx)
	{
		label_incr("set_data");
		std::stringstream ss;
		ss << idx;
		set_label("set_data", ss.str());
	
		std::memcpy(&result_[0], data.get(), 16);
		type_ = type;
		shape_ = shape;
	}

	std::string expect_result_;
	std::string result_;
	TENS_TYPE type_ = BAD_T;
	nnet::tensorshape shape_;
};


// cover tensor: constructor, get_shape, get_type
TEST_F(TENSOR, Constructor_C000)
{
	nnet::tensorshape exshape(random_shape(this));
	nnet::tensor ten(exshape);
	nnet::tensorshape gotshape = ten.get_shape();

	EXPECT_TRUE(tensorshape_equal(exshape, gotshape)) <<
		testutils::sprintf("expecting %p, got %p", &exshape, &gotshape);
	EXPECT_EQ(BAD_T, ten.get_type());
}


// cover tensor: read_from(const idata_src& src), write_to, get_type, has_data
TEST_F(TENSOR, ReadFrom_C001)
{
	nnet::tensorshape pshape(random_undef_shape(this));
	nnet::tensorshape cshape(random_def_shape(this));
	nnet::tensor incom(pshape);
	nnet::tensor comp(cshape);
	std::stringstream ss;
	print_shape(cshape, ss);

	mock_data_src src(this);
	EXPECT_FALSE(incom.read_from(src)) <<
		testutils::sprintf("tensor with shape %p successfully read from src", &pshape);
	EXPECT_EQ(BAD_T, incom.get_type());
	EXPECT_FALSE(incom.has_data()) <<
		testutils::sprintf("tensor with shape %p has data", &pshape);
	testify::mocker::EXPECT_CALL(&src, "get_data", 0);

	EXPECT_TRUE(comp.read_from(src)) <<
		testutils::sprintf("tensor with shape %p failed to read from src", &cshape);
	EXPECT_EQ(src.type_, comp.get_type());
	EXPECT_TRUE(comp.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't has data", &cshape);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&src, "get_data", 1)) <<
		"expecting src::get_data to be called once";
	EXPECT_TRUE(testify::mocker::EXPECT_VALUE(&src, "get_data", ss.str())) <<
		testutils::sprintf("expecting src::get_data to be called with shape %p", &cshape);

	size_t idx =	get_int(1, "idx")[0];
	std::stringstream ss2;
	ss2 << idx;
	mock_data_dest dest(src);

	EXPECT_THROW(incom.write_to(dest, idx), std::exception);
	testify::mocker::EXPECT_CALL(&dest, "set_data", 0);

	comp.write_to(dest, idx);
	EXPECT_STREQ(dest.expect_result_.c_str(), dest.result_.c_str());
	EXPECT_EQ(src.type_, dest.type_);
	EXPECT_TRUE(tensorshape_equal(cshape, dest.shape_)) <<
		testutils::sprintf("expect shape %p, got %p", &cshape, &dest.shape_);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&dest, "set_data", 1)) <<
		"expecting dest::set_data to be called once";
	EXPECT_TRUE(testify::mocker::EXPECT_VALUE(&dest, "set_data", ss2.str())) <<
		testutils::sprintf("expecting dest::set_data to be called with index %d", idx);
}


// cover tensor: read_from(const idata_src& src, const tensorshape shape), get_type, has_data
TEST_F(TENSOR, ReadFromShape_C001)
{
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> goodlist = random_def_shape(this);
	std::vector<size_t> plist = make_partial(this, goodlist);
	std::vector<size_t> badlist = make_incompatible(clist);

	nnet::tensorshape pshape(plist);
	nnet::tensorshape good_shape(goodlist);

	nnet::tensorshape cshape(clist);
	nnet::tensorshape bad_shape(badlist);
	
	nnet::tensor incom(pshape);
	nnet::tensor comp(cshape);
	nnet::tensor incom2(pshape);
	nnet::tensor comp2(cshape);
	std::stringstream goodstream;
	print_shape(good_shape, goodstream);
	std::stringstream cstream;
	print_shape(cshape, cstream);

	mock_data_src src(this);
	// complete shape input for partial allowed
	EXPECT_TRUE(incom.read_from(src, good_shape)) <<
		testutils::sprintf("tensor failed to read using shape %p", &good_shape);
	EXPECT_EQ(src.type_, incom.get_type());
	EXPECT_TRUE(incom.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &good_shape);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&src, "get_data", 1)) <<
		"expecting src::get_data to be called once";
	EXPECT_TRUE(testify::mocker::EXPECT_VALUE(&src, "get_data", goodstream.str())) <<
		testutils::sprintf("expecting src::get_data to be called with shape %p", &good_shape);

	// complete shape input for complete allowed
	EXPECT_TRUE(comp.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read using shape %p", &cshape);
	EXPECT_EQ(src.type_, comp.get_type());
	EXPECT_TRUE(comp.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&src, "get_data", 2)) <<
		"expecting src::get_data to be called twice";
	EXPECT_TRUE(testify::mocker::EXPECT_VALUE(&src, "get_data", cstream.str())) <<
		testutils::sprintf("expecting src::get_data to be called with shape %p", &cshape);

	// partial shape input
	EXPECT_FALSE(incom2.read_from(src, pshape)) <<
		testutils::sprintf("tensor successfully read using shape %p", &pshape);
	EXPECT_EQ(BAD_T, incom2.get_type());
	EXPECT_FALSE(incom2.has_data()) <<
		testutils::sprintf("tensor read using shape %p have data", &pshape);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&src, "get_data", 2)) <<
		"expecting src::get_data to be called twice";

	// incompatible shape input
	EXPECT_FALSE(comp2.read_from(src, bad_shape)) <<
		testutils::sprintf("tensor successfully read using shape %p", &bad_shape);
	EXPECT_EQ(BAD_T, comp2.get_type());
	EXPECT_FALSE(comp2.has_data()) <<
		testutils::sprintf("tensor read using shape %p have data", &bad_shape);
	EXPECT_TRUE(testify::mocker::EXPECT_CALL(&src, "get_data", 2)) <<
		"expecting src::get_data to be called twice";
}


// cover tensor:
// clone and assignment
TEST_F(TENSOR, Copy_C002)
{
	std::vector<size_t> empty;
	nnet::tensor undefassign(empty);
	nnet::tensor incomassign(empty);
	nnet::tensor compassign(empty);

	nnet::tensorshape pshape = random_undef_shape(this);
	nnet::tensorshape cshape = random_def_shape(this);

	nnet::tensor undef(empty);
	nnet::tensor incom(pshape);
	nnet::tensor comp(cshape);

	// initialize
	mock_data_src src(this);
	ASSERT_TRUE(undef.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read from src with shape %p", &cshape);
	ASSERT_TRUE(comp.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read from src with shape %p", &cshape);

	nnet::tensor undefcpy(undef);
	nnet::tensor incomcpy(incom);
	nnet::tensor compcpy(comp);

	nnet::tensorshape expectc = undefcpy.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc);
	nnet::tensorshape expectp = incomcpy.get_shape();
	ASSERT_TRUE(tensorshape_equal(pshape, expectp)) <<
		testutils::sprintf("expecting shape %p, got %p", &pshape, &expectp);
	nnet::tensorshape expectc2 = compcpy.get_shape();
	EXPECT_TRUE(tensorshape_equal(cshape, expectc2)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc2);

	mock_data_dest undefcpy_dest(src);
	mock_data_dest compcpy_dest(src);
	undefcpy.write_to(undefcpy_dest);
	compcpy.write_to(compcpy_dest);

	EXPECT_EQ(src.type_, undefcpy.get_type());
	ASSERT_TRUE(undefcpy.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(undefcpy_dest.expect_result_.c_str(), undefcpy_dest.result_.c_str());

	EXPECT_EQ(BAD_T, incomcpy.get_type());
	EXPECT_FALSE(incomcpy.has_data()) <<
		"incompcpy has data, expecting no data";

	EXPECT_EQ(src.type_, compcpy.get_type());
	ASSERT_TRUE(compcpy.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(compcpy_dest.expect_result_.c_str(), compcpy_dest.result_.c_str());

	undefassign = undef;
	incomassign = incom;
	compassign = comp;

	nnet::tensorshape expectc3 = undefassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc3)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc3);
	nnet::tensorshape expectp2 = incomassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(pshape, expectp2)) <<
		testutils::sprintf("expecting shape %p, got %p", &pshape, &expectp2);
	nnet::tensorshape expectc4 = compassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc4)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc4);

	mock_data_dest undefassign_dest(src);
	mock_data_dest compassign_dest(src);
	undefassign.write_to(undefassign_dest);
	compassign.write_to(compassign_dest);

	EXPECT_EQ(src.type_, undefassign.get_type());
	ASSERT_TRUE(undefassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(undefassign_dest.expect_result_.c_str(), undefassign_dest.result_.c_str());

	EXPECT_EQ(BAD_T, incomassign.get_type());
	EXPECT_FALSE(incomassign.has_data()) <<
		"incomassign has data, expecting no data";

	EXPECT_EQ(src.type_, compassign.get_type());
	ASSERT_TRUE(compassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(compassign_dest.expect_result_.c_str(), compassign_dest.result_.c_str());
}


// cover tensor:
// move constructor and assignment
TEST_F(TENSOR, Move_C002)
{
	std::vector<size_t> empty;
	nnet::tensor undefassign(empty);
	nnet::tensor incomassign(empty);
	nnet::tensor compassign(empty);

	nnet::tensorshape pshape = random_undef_shape(this);
	nnet::tensorshape cshape = random_def_shape(this);

	nnet::tensor undef(empty);
	nnet::tensor incom(pshape);
	nnet::tensor comp(cshape);

	// initialize
	mock_data_src src(this);
	ASSERT_TRUE(undef.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read from src with shape %p", &cshape);
	ASSERT_TRUE(comp.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read from src with shape %p", &cshape);

	nnet::tensor undefmv(std::move(undef));
	nnet::tensor incommv(std::move(incom));
	nnet::tensor compmv(std::move(comp));

	EXPECT_FALSE(undef.has_data()) <<
		"moved undef has data";
	EXPECT_FALSE(incom.has_data()) <<
		"moved incom has data";
	EXPECT_FALSE(comp.has_data()) <<
		"moved comp has data";
	EXPECT_EQ(BAD_T, undef.get_type());
	EXPECT_EQ(BAD_T, incom.get_type());
	EXPECT_EQ(BAD_T, comp.get_type());
	nnet::tensorshape s0 = undef.get_shape();
	nnet::tensorshape s1 = incom.get_shape();
	nnet::tensorshape s2 = comp.get_shape();
	EXPECT_FALSE(s0.is_part_defined()) <<
		testutils::sprintf("moved undef has defined shape %p", &s0);
	EXPECT_FALSE(s1.is_part_defined()) <<
		testutils::sprintf("moved incom has defined shape %p", &s1);
	EXPECT_FALSE(s2.is_part_defined()) <<
		testutils::sprintf("moved comp has defined shape %p", &s2);

	nnet::tensorshape expectc = undefmv.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc);
	nnet::tensorshape expectp = incommv.get_shape();
	ASSERT_TRUE(tensorshape_equal(pshape, expectp)) <<
		testutils::sprintf("expecting shape %p, got %p", &pshape, &expectp);
	nnet::tensorshape expectc2 = compmv.get_shape();
	EXPECT_TRUE(tensorshape_equal(cshape, expectc2)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc2);

	mock_data_dest undefmv_dest(src);
	mock_data_dest compmv_dest(src);
	undefmv.write_to(undefmv_dest);
	compmv.write_to(compmv_dest);

	EXPECT_EQ(src.type_, undefmv.get_type());
	ASSERT_TRUE(undefmv.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(undefmv_dest.expect_result_.c_str(), undefmv_dest.result_.c_str());

	EXPECT_EQ(BAD_T, incommv.get_type());
	EXPECT_FALSE(incommv.has_data()) <<
		"incompmv has data, expecting no data";

	EXPECT_EQ(src.type_, compmv.get_type());
	ASSERT_TRUE(compmv.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(compmv_dest.expect_result_.c_str(), compmv_dest.result_.c_str());

	undefassign = std::move(undefmv);
	incomassign = std::move(incommv);
	compassign = std::move(compmv);

	EXPECT_FALSE(undefmv.has_data()) <<
		"moved undef has data";
	EXPECT_FALSE(incommv.has_data()) <<
		"moved incom has data";
	EXPECT_FALSE(compmv.has_data()) <<
		"moved comp has data";
	EXPECT_EQ(BAD_T, undefmv.get_type());
	EXPECT_EQ(BAD_T, incommv.get_type());
	EXPECT_EQ(BAD_T, compmv.get_type());
	nnet::tensorshape s3 = undefmv.get_shape();
	nnet::tensorshape s4 = incommv.get_shape();
	nnet::tensorshape s5 = compmv.get_shape();
	EXPECT_FALSE(s3.is_part_defined()) <<
		testutils::sprintf("moved undefmv has defined shape %p", &s3);
	EXPECT_FALSE(s4.is_part_defined()) <<
		testutils::sprintf("moved incommv has defined shape %p", &s4);
	EXPECT_FALSE(s5.is_part_defined()) <<
		testutils::sprintf("moved compmv has defined shape %p", &s5);

	nnet::tensorshape expectc3 = undefassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc3)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc3);
	nnet::tensorshape expectp2 = incomassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(pshape, expectp2)) <<
		testutils::sprintf("expecting shape %p, got %p", &pshape, &expectp2);
	nnet::tensorshape expectc4 = compassign.get_shape();
	ASSERT_TRUE(tensorshape_equal(cshape, expectc4)) <<
		testutils::sprintf("expecting shape %p, got %p", &cshape, &expectc4);

	mock_data_dest undefassign_dest(src);
	mock_data_dest compassign_dest(src);
	undefassign.write_to(undefassign_dest);
	compassign.write_to(compassign_dest);

	EXPECT_EQ(src.type_, undefassign.get_type());
	ASSERT_TRUE(undefassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(undefassign_dest.expect_result_.c_str(), undefassign_dest.result_.c_str());

	EXPECT_EQ(BAD_T, incomassign.get_type());
	EXPECT_FALSE(incomassign.has_data()) <<
		"incomassign has data, expecting no data";

	EXPECT_EQ(src.type_, compassign.get_type());
	ASSERT_TRUE(compassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(compassign_dest.expect_result_.c_str(), compassign_dest.result_.c_str());
}



// cover tensor:
// get_shape, n_elems
TEST_F(TENSOR, ShapeAccessor_C003)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);

	nnet::tensorshape pshape = pds;
	nnet::tensorshape cshape = cds;
	nnet::tensor ten(pshape);

	nnet::tensorshape expshape = ten.get_shape();
	size_t exzero = ten.n_elems();
	size_t prank = ten.rank();
	std::vector<size_t> plist = ten.dims();
	EXPECT_TRUE(tensorshape_equal(pshape, expshape)) <<
		testutils::sprintf("expect shape %p, got %p", &pshape, &expshape);
	EXPECT_EQ(0, exzero);
	EXPECT_EQ(pshape.rank(), prank);
	EXPECT_TRUE(std::equal(pds.begin(), pds.end(), plist.begin())) <<
		testutils::sprintf("expect %vd, got %vd", &pds, &plist);

	mock_data_src src(this);
	ASSERT_TRUE(ten.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read from src with shape %p", &pshape);

	nnet::tensorshape excshape = ten.get_shape();
	size_t celems = ten.n_elems();
	size_t crank = ten.rank();
	std::vector<size_t> clist = ten.dims();
	EXPECT_TRUE(tensorshape_equal(cshape, excshape)) <<
		testutils::sprintf("expect shape %p, got %p", &cshape, &excshape);
	EXPECT_EQ(cshape.n_elems(), celems);
	EXPECT_EQ(cshape.rank(), crank);
	EXPECT_TRUE(std::equal(cds.begin(), cds.end(), clist.begin())) <<
		testutils::sprintf("expect %vd, got %vd", &cds, &clist);
}


/*
// cover tensor: is_same_size
TEST_F(tensor, IsSameSize_C004)
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


// cover tensor:
// is_compatible_with tensor => bool is_compatible_with (const tensor& other) const
TEST_F(tensor, IsCompatibleWithTensor_C005)
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


// cover tensor:
// is_compatible_with vector => bool is_compatible_with (size_t ndata) const,
// is_loosely_compatible_with
TEST_F(tensor, IsCompatibleWithVector_C006)
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
TEST_F(tensor, GuessShape_C007)
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


// cover tensor: expose
TEST_F(TENSOR, Expose_C008)
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


// cover tensor: has_data
TEST_F(TENSOR, HasData_C009)
{
	asdf
}


// cover tensor: total_bytes
TEST_F(TENSOR, TotalBytes_C010)
{
	asdf
}


// cover tensor: clear
TEST_F(TENSOR, Clear_C011)
{
	asdf
}


// cover tensor: set_shape
TEST_F(TENSOR, SetShape_C012)
{
	asdf
}


// cover tensor: copy_from
TEST_F(TENSOR, CopyFrom_C013)
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
		std::vector<size_t> incoord = cshape.coord_from_idx(i);
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
		std::vector<size_t> incoord = cshape2.coord_from_idx(i);
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


// cover tensor: serialize
TEST_F(TENSOR, Serialize_C014)
{
}


// cover tensor: from_proto
TEST_F(TENSOR, FromProto_C015)
{
}
*/


#endif /* DISABLE_TENSOR_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */

