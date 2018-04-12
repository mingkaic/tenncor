//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"
#include "mock_src.hpp"
#include "mock_dest.hpp"

#include "tensor/tensor.hpp"


#ifndef DISABLE_TENSOR_TEST


class TENSOR : public testutils::fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutils::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


using namespace testutils;


// cover tensor: constructor, get_shape, get_type
TEST_F(TENSOR, Constructor_C000)
{
	nnet::tensorshape exshape(random_shape(this));
	nnet::tensor ten(exshape);
	nnet::tensorshape gotshape = ten.get_shape();

	EXPECT_SHAPEQ(exshape,  gotshape);
	EXPECT_EQ(nnet::BAD_T, ten.get_type());
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
	EXPECT_EQ(nnet::BAD_T, incom.get_type());
	EXPECT_FALSE(incom.has_data()) <<
		testutils::sprintf("tensor with shape %p has data", &pshape);
	size_t sgetdata = testify::mocker::get_usage(&src, "get_data");
	EXPECT_EQ(0, sgetdata);

	EXPECT_TRUE(comp.read_from(src)) <<
		testutils::sprintf("tensor with shape %p failed to read from src", &cshape);
	EXPECT_TRUE(comp.read_from(src)) <<
		testutils::sprintf("second read on tensor with shape %p failed", &cshape);
	EXPECT_EQ(src.type_, comp.get_type());
	EXPECT_TRUE(comp.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't has data", &cshape);

	sgetdata = testify::mocker::get_usage(&src, "get_data");
	optional<std::string> sgetval = testify::mocker::get_value(&src, "get_data");
	ASSERT_TRUE((bool) sgetval) <<
		"no label get_data for src";
	EXPECT_EQ(2, sgetdata);
	EXPECT_STREQ(ss.str().c_str(), sgetval->c_str());

	size_t idx = get_int(1, "idx")[0];
	std::stringstream ss2;
	ss2 << idx;
	mock_data_dest dest;

	EXPECT_THROW(incom.write_to(dest, idx), std::exception);
	size_t dsetdata = testify::mocker::get_usage(&dest, "set_data");
	EXPECT_EQ(0, dsetdata);

	comp.write_to(dest, idx);
	EXPECT_STREQ(src.uuid_.c_str(), dest.result_.c_str());
	EXPECT_EQ(src.type_, dest.type_);
	EXPECT_SHAPEQ(cshape,  dest.shape_);

	dsetdata = testify::mocker::get_usage(&dest, "set_data");
	optional<std::string> dsetval = testify::mocker::get_value(&dest, "set_data");
	ASSERT_TRUE((bool) dsetval) <<
		"no label set_data for dest";
	EXPECT_EQ(1, dsetdata);
	EXPECT_STREQ(ss2.str().c_str(), dsetval->c_str());
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
	EXPECT_TRUE(incom.read_from(src, good_shape)) <<
		testutils::sprintf("second read on tensor failed using shape %p", &good_shape);
	EXPECT_EQ(src.type_, incom.get_type());
	EXPECT_TRUE(incom.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &good_shape);

	size_t sgetdata = testify::mocker::get_usage(&src, "get_data");
	optional<std::string> sgetval = testify::mocker::get_value(&src, "get_data");
	ASSERT_TRUE((bool) sgetval) <<
		"no label get_data for src";
	EXPECT_EQ(2, sgetdata);
	EXPECT_STREQ(goodstream.str().c_str(), sgetval->c_str());

	// complete shape input for complete allowed
	EXPECT_TRUE(comp.read_from(src, cshape)) <<
		testutils::sprintf("tensor failed to read using shape %p", &cshape);
	EXPECT_TRUE(comp.read_from(src, cshape)) <<
		testutils::sprintf("second read on tensor failed using shape %p", &cshape);
	EXPECT_EQ(src.type_, comp.get_type());
	EXPECT_TRUE(comp.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);

	sgetdata = testify::mocker::get_usage(&src, "get_data");
	sgetval = testify::mocker::get_value(&src, "get_data");
	ASSERT_TRUE((bool) sgetval) <<
		"no label get_data for src";
	EXPECT_EQ(4, sgetdata);
	EXPECT_STREQ(cstream.str().c_str(), sgetval->c_str());

	// partial shape input
	EXPECT_FALSE(incom2.read_from(src, pshape)) <<
		testutils::sprintf("tensor successfully read using shape %p", &pshape);
	EXPECT_EQ(nnet::BAD_T, incom2.get_type());
	EXPECT_FALSE(incom2.has_data()) <<
		testutils::sprintf("tensor read using shape %p have data", &pshape);

	sgetdata = testify::mocker::get_usage(&src, "get_data");
	EXPECT_EQ(4, sgetdata);

	// incompatible shape input
	EXPECT_FALSE(comp2.read_from(src, bad_shape)) <<
		testutils::sprintf("tensor successfully read using shape %p", &bad_shape);
	EXPECT_EQ(nnet::BAD_T, comp2.get_type());
	EXPECT_FALSE(comp2.has_data()) <<
		testutils::sprintf("tensor read using shape %p have data", &bad_shape);

	sgetdata = testify::mocker::get_usage(&src, "get_data");
	EXPECT_EQ(4, sgetdata);
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
	EXPECT_SHAPEQ(cshape,  expectc2);

	mock_data_dest undefcpy_dest;
	mock_data_dest compcpy_dest;
	undefcpy.write_to(undefcpy_dest);
	compcpy.write_to(compcpy_dest);

	EXPECT_EQ(src.type_, undefcpy.get_type());
	ASSERT_TRUE(undefcpy.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), undefcpy_dest.result_.c_str());

	EXPECT_EQ(nnet::BAD_T, incomcpy.get_type());
	EXPECT_FALSE(incomcpy.has_data()) <<
		"incompcpy has data, expecting no data";

	EXPECT_EQ(src.type_, compcpy.get_type());
	ASSERT_TRUE(compcpy.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), compcpy_dest.result_.c_str());

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

	mock_data_dest undefassign_dest;
	mock_data_dest compassign_dest;
	undefassign.write_to(undefassign_dest);
	compassign.write_to(compassign_dest);

	EXPECT_EQ(src.type_, undefassign.get_type());
	ASSERT_TRUE(undefassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), undefassign_dest.result_.c_str());

	EXPECT_EQ(nnet::BAD_T, incomassign.get_type());
	EXPECT_FALSE(incomassign.has_data()) <<
		"incomassign has data, expecting no data";

	EXPECT_EQ(src.type_, compassign.get_type());
	ASSERT_TRUE(compassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), compassign_dest.result_.c_str());
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
	EXPECT_EQ(nnet::BAD_T, undef.get_type());
	EXPECT_EQ(nnet::BAD_T, incom.get_type());
	EXPECT_EQ(nnet::BAD_T, comp.get_type());
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
	EXPECT_SHAPEQ(cshape,  expectc2);
	

	mock_data_dest undefmv_dest;
	mock_data_dest compmv_dest;
	undefmv.write_to(undefmv_dest);
	compmv.write_to(compmv_dest);

	EXPECT_EQ(src.type_, undefmv.get_type());
	ASSERT_TRUE(undefmv.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), undefmv_dest.result_.c_str());

	EXPECT_EQ(nnet::BAD_T, incommv.get_type());
	EXPECT_FALSE(incommv.has_data()) <<
		"incompmv has data, expecting no data";

	EXPECT_EQ(src.type_, compmv.get_type());
	ASSERT_TRUE(compmv.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), compmv_dest.result_.c_str());

	undefassign = std::move(undefmv);
	incomassign = std::move(incommv);
	compassign = std::move(compmv);

	EXPECT_FALSE(undefmv.has_data()) <<
		"moved undef has data";
	EXPECT_FALSE(incommv.has_data()) <<
		"moved incom has data";
	EXPECT_FALSE(compmv.has_data()) <<
		"moved comp has data";
	EXPECT_EQ(nnet::BAD_T, undefmv.get_type());
	EXPECT_EQ(nnet::BAD_T, incommv.get_type());
	EXPECT_EQ(nnet::BAD_T, compmv.get_type());
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

	mock_data_dest undefassign_dest;
	mock_data_dest compassign_dest;
	undefassign.write_to(undefassign_dest);
	compassign.write_to(compassign_dest);

	EXPECT_EQ(src.type_, undefassign.get_type());
	ASSERT_TRUE(undefassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), undefassign_dest.result_.c_str());

	EXPECT_EQ(nnet::BAD_T, incomassign.get_type());
	EXPECT_FALSE(incomassign.has_data()) <<
		"incomassign has data, expecting no data";

	EXPECT_EQ(src.type_, compassign.get_type());
	ASSERT_TRUE(compassign.has_data()) <<
		testutils::sprintf("tensor read using shape %p doesn't have data", &cshape);
	EXPECT_STREQ(src.uuid_.c_str(), compassign_dest.result_.c_str());
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
	EXPECT_SHAPEQ(pshape,  expshape);
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
	EXPECT_SHAPEQ(cshape,  excshape);
	EXPECT_EQ(cshape.n_elems(), celems);
	EXPECT_EQ(cshape.rank(), crank);
	EXPECT_TRUE(std::equal(cds.begin(), cds.end(), clist.begin())) <<
		testutils::sprintf("expect %vd, got %vd", &cds, &clist);
}


// cover tensor: is_same_size
TEST_F(TENSOR, IsSameSize_C004)
{
	std::vector<size_t> empty;
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> ilist = make_incompatible(clist); // not same as cshape
	std::vector<size_t> plist;
	std::vector<size_t> badpartial;
	make_incom_partials(this, clist, plist, badpartial); // same as cshape
	nnet::tensorshape cshape = clist;
	nnet::tensorshape pshape = plist;
	nnet::tensorshape badpshape = badpartial;
	nnet::tensorshape badshape = ilist;
	nnet::tensor bad(badshape);
	nnet::tensor undef(empty);
	nnet::tensor pcom(pshape);
	nnet::tensor badp(badpshape);
	nnet::tensor comp(cshape);

	// allowed compatible
	// pcom, undef are both unallocated
	ASSERT_FALSE(undef.has_data()) << 
		"tensor with empty shape has data";
	ASSERT_FALSE(pcom.has_data()) <<
		testutils::sprintf("tensor with shape %p has data", &pshape);

	// undef is same as anything
	EXPECT_TRUE(undef.is_same_size(bad)) <<
		testutils::sprintf("undef not compatible with shape %p", &badshape);
	EXPECT_TRUE(undef.is_same_size(comp)) <<
		testutils::sprintf("undef not compatible with shape %p", &cshape);
	EXPECT_TRUE(undef.is_same_size(pcom)) <<
		testutils::sprintf("undef not compatible with shape %p", &pshape);
	// pcom is same as comp, but not bad or scalar
	EXPECT_TRUE(pcom.is_same_size(comp)) <<
		testutils::sprintf("partial with shape %p not compatible with shape %p", &pshape, &cshape);
	EXPECT_FALSE(pcom.is_same_size(bad)) <<
		testutils::sprintf("partial with shape %p is compatible with bad shape %p", &pshape, &badshape);

	// trimmed compatible
	// padd clist
	std::vector<size_t> npads = get_int(4, "npads", {3, 17});
	nnet::tensorshape p1 = padd(clist, npads[0], npads[1]); // same
	nnet::tensorshape p2 = padd(clist, npads[2], npads[3]); // same
	clist.push_back(2);
	nnet::tensorshape p3 = padd(clist, npads[2], npads[3]); // not same
	nnet::tensor comp2(p1);
	nnet::tensor comp3(p2);
	nnet::tensor bad2(p3);

	mock_data_src src(this);
	ASSERT_TRUE(comp2.read_from(src)) <<
		testutils::sprintf("tensor with shape %p successfully read from src", &p1);
	ASSERT_TRUE(comp3.read_from(src)) <<
		testutils::sprintf("tensor with shape %p successfully read from src", &p2);
	ASSERT_TRUE(pcom.read_from(src, cshape)) <<
		testutils::sprintf("tensor with shape %p successfully read from src", &cshape);
	ASSERT_TRUE(bad.read_from(src)) <<
		testutils::sprintf("tensor with shape %p successfully read from src", &p3);

	ASSERT_TRUE(comp2.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't have data", &p1);
	ASSERT_TRUE(comp3.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't have data", &p2);
	ASSERT_TRUE(pcom.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't have data", &pshape);
	ASSERT_TRUE(bad.has_data()) <<
		testutils::sprintf("tensor with shape %p doesn't have data", &badshape);

	EXPECT_TRUE(comp.is_same_size(comp2)) << 
		testutils::sprintf("uninit with shape %p is same size as init with shape %p", &cshape, &p1);
	EXPECT_TRUE(comp.is_same_size(comp3)) << 
		testutils::sprintf("uninit with shape %p is same size as init with shape %p", &cshape, &p2);
	EXPECT_TRUE(comp2.is_same_size(comp3)) <<
		testutils::sprintf("init with shapes %p and %p are not the same size", &p1, &p2);
	EXPECT_FALSE(badp.is_same_size(pcom)) << 
		testutils::sprintf("uninit with shape %p is same size as init with allowed shape %p", &badpshape, &pshape);

	EXPECT_FALSE(comp.is_same_size(bad)) << 
		testutils::sprintf("uninit with shape %p is same size as bad with shape %p", &cshape, &badshape);
	EXPECT_FALSE(comp2.is_same_size(bad)) << 
		testutils::sprintf("init with shape %p is same size as bad init with shape %p", &p1, &badshape);
	EXPECT_FALSE(comp3.is_same_size(bad)) << 
		testutils::sprintf("init with shape %p is same size as bad init with shape %p", &p2, &badshape);

	EXPECT_FALSE(comp.is_same_size(bad2)) << 
		testutils::sprintf("uninit with shape %p is same size as bad init with shape %p", &cshape, &p3);
	EXPECT_FALSE(comp2.is_same_size(bad2)) << 
		testutils::sprintf("init with shape %p is same size as bad init with shape %p", &p1, &p3);
	EXPECT_FALSE(comp3.is_same_size(bad2)) << 
		testutils::sprintf("init with shape %p is same size as bad init with shape %p", &p2, &p3);
}


// cover tensor:
// is_compatible_with tensor => bool is_compatible_with (const tensor& other) const
TEST_F(TENSOR, IsCompatibleWithTensor_C005)
{
	std::vector<size_t> empty;
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> ilist = make_incompatible(clist); // not same as cshape
	std::vector<size_t> plist = make_partial(this, clist); // same as cshape
	nnet::tensorshape cshape = clist;
	nnet::tensorshape badshape = ilist;
	nnet::tensorshape pshape = plist;
	nnet::tensor undef(empty);
	nnet::tensor comp(cshape);
	nnet::tensor bad(badshape);
	nnet::tensor pcom(pshape);

	// undefined tensor is compatible with anything
	EXPECT_TRUE(undef.is_compatible_with(undef)) << 
		"empty shape is incompatible with empty shape";
	EXPECT_TRUE(undef.is_compatible_with(comp)) <<
		testutils::sprintf("empty shape is incompatible with shape %p", &cshape);
	EXPECT_TRUE(undef.is_compatible_with(pcom)) <<
		testutils::sprintf("empty shape is incompatible with shape %p", &pshape);
	EXPECT_TRUE(undef.is_compatible_with(bad)) <<
		testutils::sprintf("empty shape is incompatible with shape %p", &badshape);

	EXPECT_TRUE(pcom.is_compatible_with(comp)) <<
		testutils::sprintf("shape %p is compatible with shape %p", &pshape, &cshape);
	EXPECT_TRUE(pcom.is_compatible_with(pcom)) <<
		testutils::sprintf("shape %p is incompatible with itself", &pshape);
	EXPECT_FALSE(pcom.is_compatible_with(bad)) <<
		testutils::sprintf("bad shape %p is compatible with shape %p", &badshape, &pshape);

	EXPECT_FALSE(bad.is_compatible_with(comp)) <<
		testutils::sprintf("bad shape %p is compatible with shape %p", &badshape, &cshape);
}


// cover tensor:
// is_compatible_with vector => bool is_compatible_with (size_t ndata) const,
// is_loosely_compatible_with
TEST_F(TENSOR, IsCompatibleWithNData_C006)
{
	std::vector<size_t> empty;
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist = make_partial(this, clist); // same as cshape
	nnet::tensorshape pshape = plist;
	nnet::tensorshape cshape = clist;

	nnet::tensor undef(empty);
	nnet::tensor comp(cshape);
	nnet::tensor pcom(pshape);

	size_t exactdata = cshape.n_elems();
	size_t lowerdata = 1;
	if (exactdata >= 3)
	{
		lowerdata = exactdata - get_int(1,
			"exactdata - lowerdata", {1, exactdata-1})[0];
	}
	size_t upperdata = exactdata + get_int(1,
		"upperdata - exactdata", {1, exactdata-1})[0];

	EXPECT_TRUE(comp.is_compatible_with(exactdata)) <<
		testutils::sprintf("shape %p is incompatible with nelem=%d", &cshape, exactdata);
	EXPECT_FALSE(comp.is_compatible_with(lowerdata)) <<
		testutils::sprintf("shape %p is compatible with nelem=%d", &cshape, lowerdata);
	EXPECT_FALSE(comp.is_compatible_with(upperdata)) <<
		testutils::sprintf("shape %p is compatible with nelem=%d", &cshape, upperdata);

	EXPECT_TRUE(comp.is_loosely_compatible_with(exactdata)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &cshape, exactdata);
	EXPECT_TRUE(comp.is_loosely_compatible_with(lowerdata)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &cshape, lowerdata);
	EXPECT_FALSE(comp.is_loosely_compatible_with(upperdata)) <<
		testutils::sprintf("shape %p is loosely compatible with nelem=%d", &cshape, upperdata);

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

	EXPECT_TRUE(pcom.is_compatible_with(exactdata2)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, exactdata2);
	EXPECT_TRUE(pcom.is_compatible_with(moddata)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, moddata);
	EXPECT_FALSE(pcom.is_compatible_with(lowerdata2)) <<
		testutils::sprintf("shape %p is loosely compatible with nelem=%d", &pshape, lowerdata2);
	EXPECT_FALSE(pcom.is_compatible_with(upperdata2)) <<
		testutils::sprintf("shape %p is loosely compatible with nelem=%d", &pshape, upperdata2);

	EXPECT_TRUE(pcom.is_loosely_compatible_with(exactdata2)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, exactdata2);
	EXPECT_TRUE(pcom.is_loosely_compatible_with(moddata)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, moddata);
	EXPECT_TRUE(pcom.is_loosely_compatible_with(lowerdata2)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, lowerdata2);
	EXPECT_TRUE(pcom.is_loosely_compatible_with(upperdata2)) <<
		testutils::sprintf("shape %p is loosely incompatible with nelem=%d", &pshape, upperdata2);

	// undef is compatible with everything
	EXPECT_TRUE(undef.is_compatible_with(exactdata)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", exactdata);
	EXPECT_TRUE(undef.is_compatible_with(exactdata2)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", exactdata2);
	EXPECT_TRUE(undef.is_compatible_with(lowerdata)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", lowerdata);
	EXPECT_TRUE(undef.is_compatible_with(lowerdata2)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", lowerdata2);
	EXPECT_TRUE(undef.is_compatible_with(upperdata)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", upperdata);
	EXPECT_TRUE(undef.is_compatible_with(upperdata2)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", upperdata2);
	EXPECT_TRUE(undef.is_compatible_with(moddata)) <<
		testutils::sprintf("undef shape is incompatible with nelem=%d", moddata);

	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", exactdata);
	EXPECT_TRUE(undef.is_loosely_compatible_with(exactdata2)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", exactdata2);
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", lowerdata);
	EXPECT_TRUE(undef.is_loosely_compatible_with(lowerdata2)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", lowerdata2);
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", upperdata);
	EXPECT_TRUE(undef.is_loosely_compatible_with(upperdata2)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", upperdata2);
	EXPECT_TRUE(undef.is_loosely_compatible_with(moddata)) <<
		testutils::sprintf("undef shape is loosely incompatible with nelem=%d", moddata);
}


// covers tensor
// guess_shape, loosely_guess_shape
TEST_F(TENSOR, GuessShape_C007)
{
	std::vector<size_t> empty;
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist = make_partial(this, clist); // same as cshape
	nnet::tensorshape pshape = plist;
	nnet::tensorshape cshape = clist;

	nnet::tensor undef(empty);
	nnet::tensor comp(cshape);
	nnet::tensor pcom(pshape);

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
	optional<nnet::tensorshape> cres = comp.guess_shape(exactdata);
	ASSERT_TRUE((bool)cres) <<
		testutils::sprintf("shape %p failed to guess nelems=%d", &cshape, exactdata);
	EXPECT_SHAPEQ(cshape,  *cres);
	EXPECT_FALSE((bool)comp.guess_shape(lowerdata)) <<
		testutils::sprintf("shape %p guessed nelems=%d", &cshape, lowerdata);
	EXPECT_FALSE((bool)comp.guess_shape(upperdata)) <<
		testutils::sprintf("shape %p guessed nelems=%d", &cshape, upperdata);

	size_t exactdata2 = pshape.n_known();
	size_t lowerdata2 = 1;
	if (exactdata2 >= 3)
	{
		lowerdata2 = exactdata2 - get_int(1,
			"exactdata2 - lowerdata2", {1, exactdata2-1})[0];
	}
	size_t moddata = exactdata2 * get_int(1, "moddata / exactdata2", {2, 15})[0];
	size_t upperdata2 = moddata + 1;

	size_t unknown = plist.size();
	for (size_t i = 0; i < plist.size(); i++)
	{
		if (0 == plist[i])
		{
			if (unknown > i)
			{
				unknown = i;
			}
			plist[i] = 1;
		}
	}
	std::vector<size_t> plist2 = plist;
	plist2[unknown] = ceil((double) moddata / (double) exactdata2);
	nnet::tensorshape pmod = plist;
	nnet::tensorshape pshape2 = plist2;
	// allowed are partially defined
	optional<nnet::tensorshape> pres = pcom.guess_shape(exactdata2);
	optional<nnet::tensorshape> pres2 = pcom.guess_shape(moddata);
	ASSERT_TRUE((bool)pres) <<
		testutils::sprintf("shape %p failed to guess nelems=%d", &pshape, exactdata2);
	ASSERT_TRUE((bool)pres2) <<
		testutils::sprintf("shape %p failed to guess nelems=%d", &pshape, moddata);
	EXPECT_SHAPEQ(pmod,  *pres);
	EXPECT_SHAPEQ(pshape2,  *pres2);
	EXPECT_FALSE((bool)pcom.guess_shape(lowerdata2)) <<
		testutils::sprintf("shape %p guessed nelems=%d", &pshape, lowerdata2);
	EXPECT_FALSE((bool)pcom.guess_shape(upperdata2)) <<
		testutils::sprintf("shape %p guessed nelems=%d", &pshape, upperdata2);

	// allowed are undefined
	optional<nnet::tensorshape> ures = undef.guess_shape(exactdata);
	optional<nnet::tensorshape> ures2 = undef.guess_shape(exactdata2);
	optional<nnet::tensorshape> ures3 = undef.guess_shape(lowerdata);
	optional<nnet::tensorshape> ures4 = undef.guess_shape(lowerdata2);
	optional<nnet::tensorshape> ures5 = undef.guess_shape(upperdata);
	optional<nnet::tensorshape> ures6 = undef.guess_shape(upperdata2);
	optional<nnet::tensorshape> ures7 = undef.guess_shape(moddata);
	ASSERT_TRUE((bool)ures) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", exactdata);
	ASSERT_TRUE((bool)ures2) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", exactdata2);
	ASSERT_TRUE((bool)ures3) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", lowerdata);
	ASSERT_TRUE((bool)ures4) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", lowerdata2);
	ASSERT_TRUE((bool)ures5) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", upperdata);
	ASSERT_TRUE((bool)ures6) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", upperdata2);
	ASSERT_TRUE((bool)ures7) <<
		testutils::sprintf("undef shape failed to guess nelems=%d", moddata);
	EXPECT_TRUE(tensorshape_equal(*ures, std::vector<size_t>({exactdata}))) <<
		testutils::sprintf("expected shape %d, got %p", exactdata, &*ures);
	EXPECT_TRUE(tensorshape_equal(*ures2, std::vector<size_t>({exactdata2}))) <<
		testutils::sprintf("expected shape %d, got %p", exactdata2, &*ures2);
	EXPECT_TRUE(tensorshape_equal(*ures3, std::vector<size_t>({lowerdata}))) <<
		testutils::sprintf("expected shape %d, got %p", lowerdata, &*ures3);
	EXPECT_TRUE(tensorshape_equal(*ures4, std::vector<size_t>({lowerdata2}))) <<
		testutils::sprintf("expected shape %d, got %p", lowerdata2, &*ures4);
	EXPECT_TRUE(tensorshape_equal(*ures5, std::vector<size_t>({upperdata}))) <<
		testutils::sprintf("expected shape %d, got %p", upperdata, &*ures5);
	EXPECT_TRUE(tensorshape_equal(*ures6, std::vector<size_t>({upperdata2}))) <<
		testutils::sprintf("expected shape %d, got %p", upperdata2, &*ures6);
	EXPECT_TRUE(tensorshape_equal(*ures7, std::vector<size_t>({moddata}))) <<
		testutils::sprintf("expected shape %d, got %p", moddata, &*ures7);
}


// cover tensor: expose
TEST_F(TENSOR, Expose_C008)
{
	mock_data_src src(this);
	nnet::tensorshape shape = std::vector<size_t>{(size_t) (16 / nnet::type_size(src.type_))};
	nnet::tensor ten(shape);
	ten.read_from(src);
	std::string result;

	// shouldn't die or throw
	switch (src.type_)
	{
		case nnet::DOUBLE:
		{
			std::vector<double> v = nnet::expose<double>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::FLOAT:
		{
			std::vector<float> v = nnet::expose<float>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::INT8:
		{
			std::vector<int8_t> v = nnet::expose<int8_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::UINT8:
		{
			std::vector<uint8_t> v = nnet::expose<uint8_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::INT16:
		{
			std::vector<int16_t> v = nnet::expose<int16_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::UINT16:
		{
			std::vector<uint16_t> v = nnet::expose<uint16_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::INT32:
		{
			std::vector<int32_t> v = nnet::expose<int32_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::UINT32:
		{
			std::vector<uint32_t> v = nnet::expose<uint32_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::INT64:
		{
			std::vector<int64_t> v = nnet::expose<int64_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		case nnet::UINT64:
		{
			std::vector<uint64_t> v = nnet::expose<uint64_t>(&ten);
			result = std::string((char*) &v[0], 16);
		}
		break;
		default:
			ASSERT_TRUE(false) << 
				"src.type_ has invalid type " << src.type_;
	}
	EXPECT_STREQ(src.uuid_.c_str(), result.c_str());
}


// cover tensor: total_bytes
TEST_F(TENSOR, TotalBytes_C009)
{
	mock_data_src src(this);
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	nnet::tensor ten(shape);
	ASSERT_FALSE(ten.has_data()) << 
		testutils::sprintf("ten w/ shape %p initialized with data", &shape);
	EXPECT_EQ(0, ten.total_bytes());

	ten.read_from(src);
	ASSERT_TRUE(ten.has_data()) << 
		testutils::sprintf("ten w/ shape %p failed to read src", &shape);
	EXPECT_EQ(shape.n_elems() * nnet::type_size(src.type_), ten.total_bytes());
}


// cover tensor: clear
TEST_F(TENSOR, Clear_C010)
{
	mock_data_src src(this);
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist = make_partial(this, clist); // same as cshape
	nnet::tensorshape cshape = clist;
	nnet::tensorshape pshape = plist;
	nnet::tensor ten(pshape);
	ASSERT_FALSE(ten.has_data()) <<
		testutils::sprintf("ten w/ shape %p initialized with data", &pshape);
	nnet::tensorshape tshape = ten.get_shape();
	EXPECT_SHAPEQ(pshape,  tshape);
	EXPECT_EQ(nnet::BAD_T, ten.get_type());

	ten.read_from(src, cshape);
	ASSERT_TRUE(ten.has_data()) <<
		testutils::sprintf("ten read with shape %p has no data", &cshape);
	tshape = ten.get_shape();
	EXPECT_SHAPEQ(cshape,  tshape);
	EXPECT_EQ(src.type_, ten.get_type());

	ten.clear();
	ASSERT_FALSE(ten.has_data()) << 
		testutils::sprintf("cleared ten w/ allowed shape %p has data", &pshape);
	tshape = ten.get_shape();
	EXPECT_SHAPEQ(pshape,  tshape);
	EXPECT_EQ(nnet::BAD_T, ten.get_type());
}


// cover tensor: set_shape
TEST_F(TENSOR, SetShape_C011)
{
	mock_data_src src(this);
	std::vector<size_t> clist = random_def_shape(this);
	std::vector<size_t> plist;
	std::vector<size_t> ilist;
	make_incom_partials(this, clist, plist, ilist);
	nnet::tensorshape eshape;
	nnet::tensorshape pshape = plist;
	nnet::tensorshape ishape = ilist;
	nnet::tensorshape cshape = clist;

	nnet::tensor undef(eshape);
	nnet::tensor comp(cshape);
	nnet::tensor pcom(pshape);
	nnet::tensor undef2(eshape);
	nnet::tensor comp2(cshape);
	nnet::tensor pcom2(pshape);
	undef.read_from(src, cshape);
	comp.read_from(src, cshape);
	pcom.read_from(src, cshape);
	undef2.read_from(src, cshape);
	comp2.read_from(src, cshape);
	pcom2.read_from(src, cshape);

	ASSERT_TRUE(undef.has_data()) <<
		testutils::sprintf("undef read with shape %p has no data", &cshape);
	ASSERT_TRUE(comp.has_data()) <<
		testutils::sprintf("comp read with shape %p has no data", &cshape);
	ASSERT_TRUE(pcom.has_data()) <<
		testutils::sprintf("pcom read with shape %p has no data", &cshape);

	// don't clear data on set_shape
	undef.set_shape(pshape);
	comp.set_shape(eshape);
	pcom.set_shape(cshape);

	EXPECT_TRUE(undef.has_data()) <<
		testutils::sprintf("undef set with shape %p has no data", &pshape);
	EXPECT_TRUE(comp.has_data()) <<
		testutils::sprintf("original %p set with empty shape has no data", &cshape);
	EXPECT_TRUE(pcom.has_data()) <<
		testutils::sprintf("original %p set with shape %p has no data", &pshape, &cshape);

	// clear and inspect allowed shape
	undef.clear();
	comp.clear();
	pcom.clear();
	nnet::tensorshape res_ushape = undef.get_shape();
	nnet::tensorshape res_cshape = comp.get_shape();
	nnet::tensorshape res_pshape = pcom.get_shape();

	EXPECT_SHAPEQ(pshape,  res_ushape);
	EXPECT_SHAPEQ(eshape,  res_cshape);
	EXPECT_SHAPEQ(cshape,  res_pshape);

	// clear data on set_shape
	undef2.set_shape(ishape);
	comp2.set_shape(ishape);
	pcom2.set_shape(ishape);

	EXPECT_FALSE(undef2.has_data()) <<
		testutils::sprintf("undef set with bad shape %p has data", &ishape);
	EXPECT_FALSE(comp2.has_data()) <<
		testutils::sprintf("original %p set with bad shape %p has data", &cshape, &ishape);
	EXPECT_FALSE(pcom2.has_data()) <<
		testutils::sprintf("original %p set with bad shape %p has data", &pshape, &ishape);

	nnet::tensorshape res_ushape2 = undef2.get_shape();
	nnet::tensorshape res_cshape2 = comp2.get_shape();
	nnet::tensorshape res_pshape2 = pcom2.get_shape();
	EXPECT_SHAPEQ(ishape,  res_ushape2);
	EXPECT_SHAPEQ(ishape,  res_cshape2);
	EXPECT_SHAPEQ(ishape,  res_pshape2);
}


// cover tensor: serialize, from_proto
TEST_F(TENSOR, Proto_C012)
{
	tenncor::TensorPb proto;

	mock_data_src src(this);
	mock_data_src src2(this);
	mock_data_src src3(this);
	nnet::tensorshape cshape = random_def_shape(this);
	nnet::tensorshape cshape2 = random_def_shape(this);
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape cshape3(clist);
	nnet::tensorshape pshape3 = make_partial(this, clist);
	nnet::tensorshape shape = random_shape(this);
	nnet::tensor comp(cshape);
	nnet::tensor comp2(cshape2);
	nnet::tensor comp3(pshape3);
	nnet::tensor ten(shape);

	// expects failure
	EXPECT_FALSE(comp.serialize(proto)) <<
		testutils::sprintf("successfully serialized uninit tensor with shape %p", &cshape);
	EXPECT_FALSE(comp2.serialize(proto)) <<
		testutils::sprintf("successfully serialized uninit tensor with shape %p", &cshape2);
	EXPECT_FALSE(ten.serialize(proto)) <<
		testutils::sprintf("successfully serialized uninit tensor with shape %p", &shape);

	comp.read_from(src);
	comp2.read_from(src2);
	comp3.read_from(src3, cshape3);

	// expects success
	EXPECT_TRUE(comp.serialize(proto)) <<
		testutils::sprintf("failed to serialized tensor with shape %p", &cshape);

	mock_data_dest dest;
	comp2.write_to(dest);
	EXPECT_STRNE(src.uuid_.c_str(), dest.result_.c_str());
	EXPECT_STREQ(src2.uuid_.c_str(), dest.result_.c_str());

	// verify data
	ten.from_proto(proto);
	comp2.from_proto(proto);

	mock_data_dest dest2;
	mock_data_dest dest3;
	ten.write_to(dest2);
	comp2.write_to(dest3);
	EXPECT_STREQ(src.uuid_.c_str(), dest2.result_.c_str());
	EXPECT_STREQ(src.uuid_.c_str(), dest3.result_.c_str());
	EXPECT_STRNE(src2.uuid_.c_str(), dest3.result_.c_str());
	EXPECT_EQ(src.type_, ten.get_type());
	EXPECT_EQ(src.type_, comp2.get_type());

	nnet::tensorshape goten = ten.get_shape();
	nnet::tensorshape gotc = comp2.get_shape();
	EXPECT_SHAPEQ(cshape,  goten);
	EXPECT_SHAPEQ(cshape,  gotc);

	// rewrite data
	ASSERT_TRUE(comp3.serialize(proto)) <<
		testutils::sprintf("failed to re-serialized tensor with shape %p", &cshape3);
	TENS_TYPE c3type = proto.type();
	std::shared_ptr<void> c3data = nnet::deserialize_data(proto.data(), c3type);
	std::string c3str((char*) c3data.get(), src3.uuid_.size());

	nnet::tensorshape c3allow(std::vector<size_t>(
		proto.allowed_shape().begin(),
		proto.allowed_shape().end()));
	nnet::tensorshape c3alloc(std::vector<size_t>(
		proto.alloced_shape().begin(),
		proto.alloced_shape().end()));

	EXPECT_STREQ(src3.uuid_.c_str(), c3str.c_str());
	EXPECT_EQ(src3.type_, c3type);
	EXPECT_SHAPEQ(pshape3,  c3allow);
	EXPECT_SHAPEQ(cshape3,  c3alloc);
}


#endif /* DISABLE_TENSOR_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */

