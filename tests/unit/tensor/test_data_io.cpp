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

#include "tensor/data_io.hpp"
#include "tensor/tensor.hpp"


#ifndef DISABLE_DIO_TEST


class DATA_IO : public testutils::fuzz_test
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


static inline void bad_op (TENS_TYPE, nnet::VARR_T, std::vector<nnet::CVAR_T>) { throw std::exception(); }


TEST_F(DATA_IO, Copy_E000)
{
	nnet::tensorshape shape1(random_def_shape(this));
	nnet::tensorshape shape2(random_def_shape(this));
	nnet::tensorshape outshape(random_def_shape(this));
	nnet::portal_dest passign;
	nnet::assign_io aassign;
	nnet::operate_io oassign(bad_op);

	nnet::portal_dest pi;
	nnet::assign_io ai;
	TENS_TYPE otype = nnet::BAD_T;
	std::vector<TENS_TYPE> otypes;
	nnet::VARR_T outdata;
	std::vector<nnet::CVAR_T> indata;
	nnet::operate_io oi(
	[&otype, &outdata, &indata](TENS_TYPE type, nnet::VARR_T dest, std::vector<nnet::CVAR_T> srcs)
	{
		otype = type;
		outdata = dest;
		indata = srcs;
	},
	[&otypes](std::vector<TENS_TYPE> types)
	{
		otypes = types;
		return types[0];
	});

	mock_data_src src1(this);
	mock_data_src src2(this);
	nnet::tensor arg1(shape1);
	nnet::tensor arg2(shape2);
	ASSERT_TRUE(arg1.read_from(src1)) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.read_from(src2)) <<
		"arg2 failed to read from src2";
	ASSERT_TRUE(arg1.has_data()) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.has_data()) <<
		"arg2 failed to read from src2";
	arg1.write_to(pi);
	arg1.write_to(ai);
	arg1.write_to(oi, 0);
	arg2.write_to(oi, 1);

	nnet::portal_dest picpy(pi);
	nnet::assign_io* aicpy = ai.clone();
	nnet::operate_io* oicpy = oi.clone();

	std::string uuid((char*) picpy.input_.data_.lock().get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, picpy.input_.type_);
	EXPECT_TRUE(tensorshape_equal(shape1, picpy.input_.shape_)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &picpy.input_.shape_);

	std::shared_ptr<void> ptr = nullptr;
	TENS_TYPE type = nnet::BAD_T;
	aicpy->get_data(ptr, type, shape1);
	uuid = std::string((char*) ptr.get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, type);

	ptr = nullptr;
	otype = nnet::BAD_T;
	oicpy->get_data(ptr, otype, outshape);
	EXPECT_EQ(outdata.first, ptr.get());
	EXPECT_TRUE(tensorshape_equal(outshape, outdata.second)) <<
		testutils::sprintf("expected %p, got %p", &outshape, &outdata.second);
	ASSERT_EQ(2, indata.size());
	uuid = std::string((char*) indata[0].first, src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape1, indata[0].second)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &indata[0].second);
	uuid = std::string((char*) indata[1].first, src2.uuid_.size());
	EXPECT_STREQ(src2.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape2, indata[1].second)) <<
		testutils::sprintf("expected %p, got %p", &shape2, &indata[1].second);
	EXPECT_EQ(src1.type_, otype);
	EXPECT_EQ(2, otypes.size());
	EXPECT_EQ(src1.type_, otypes[0]);
	EXPECT_EQ(src2.type_, otypes[1]);

	// clear
	otype = nnet::BAD_T;
	outdata = nnet::VARR_T{};
	indata.clear();
	otypes.clear();

	passign = picpy;
	aassign = *aicpy;
	oassign = *oicpy;

	uuid = std::string((char*) passign.input_.data_.lock().get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, passign.input_.type_);
	EXPECT_TRUE(tensorshape_equal(shape1, passign.input_.shape_)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &passign.input_.shape_);

	ptr = nullptr;
	type = nnet::BAD_T;
	aassign.get_data(ptr, type, shape1);
	uuid = std::string((char*) ptr.get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, type);
	
	ptr = nullptr;
	type = nnet::BAD_T;
	oassign.get_data(ptr, type, outshape);
	EXPECT_EQ(outdata.first, ptr.get());
	EXPECT_TRUE(tensorshape_equal(outshape, outdata.second)) <<
		testutils::sprintf("expected %p, got %p", &outdata, &outdata.second);
	ASSERT_EQ(2, indata.size());
	uuid = std::string((char*) indata[0].first, src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape1, indata[0].second)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &indata[0].second);
	uuid = std::string((char*) indata[1].first, src2.uuid_.size());
	EXPECT_STREQ(src2.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape2, indata[1].second)) <<
		testutils::sprintf("expected %p, got %p", &shape2, &indata[1].second);
	EXPECT_EQ(src1.type_, type);
	EXPECT_EQ(src1.type_, otype);
	EXPECT_EQ(2, otypes.size());
	EXPECT_EQ(src1.type_, otypes[0]);
	EXPECT_EQ(src2.type_, otypes[1]);

	delete aicpy;
	delete oicpy;
}


TEST_F(DATA_IO, Move_E000)
{
	nnet::tensorshape shape1(random_def_shape(this));
	nnet::tensorshape shape2(random_def_shape(this));
	nnet::tensorshape outshape(random_def_shape(this));
	nnet::portal_dest passign;
	nnet::assign_io aassign;
	nnet::operate_io oassign(bad_op);

	nnet::portal_dest pi;
	nnet::assign_io ai;
	TENS_TYPE otype = nnet::BAD_T;
	std::vector<TENS_TYPE> otypes;
	nnet::VARR_T outdata;
	std::vector<nnet::CVAR_T> indata;
	nnet::operate_io oi(
	[&otype, &outdata, &indata](TENS_TYPE type, nnet::VARR_T dest, std::vector<nnet::CVAR_T> srcs)
	{
		otype = type;
		outdata = dest;
		indata = srcs;
	},
	[&otypes](std::vector<TENS_TYPE> types)
	{
		if (types.empty()) return nnet::BAD_T;
		otypes = types;
		return types[0];
	});

	mock_data_src src1(this);
	mock_data_src src2(this);
	nnet::tensor arg1(shape1);
	nnet::tensor arg2(shape2);
	ASSERT_TRUE(arg1.read_from(src1)) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.read_from(src2)) <<
		"arg2 failed to read from src2";
	ASSERT_TRUE(arg1.has_data()) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.has_data()) <<
		"arg2 failed to read from src2";
	arg1.write_to(pi);
	arg1.write_to(ai);
	arg1.write_to(oi, 0);
	arg2.write_to(oi, 1);

	nnet::portal_dest pimv(std::move(pi));
	nnet::assign_io aimv(std::move(ai));
	nnet::operate_io oimv(std::move(oi));

	std::string uuid((char*) pimv.input_.data_.lock().get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, pimv.input_.type_);
	EXPECT_TRUE(tensorshape_equal(shape1, pimv.input_.shape_)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &pimv.input_.shape_);

	std::shared_ptr<void> ptr = nullptr;
	TENS_TYPE type = nnet::BAD_T;
	aimv.get_data(ptr, type, shape1);
	uuid = std::string((char*) ptr.get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, type);

	ptr = nullptr;
	type = nnet::BAD_T;
	oimv.get_data(ptr, type, outshape);
	EXPECT_EQ(outdata.first, ptr.get());
	EXPECT_TRUE(tensorshape_equal(outshape, outdata.second)) <<
		testutils::sprintf("expected %p, got %p", &outshape, &outdata.second);
	ASSERT_EQ(2, indata.size());
	uuid = std::string((char*) indata[0].first, src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape1, indata[0].second)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &indata[0].second);
	uuid = std::string((char*) indata[1].first, src2.uuid_.size());
	EXPECT_STREQ(src2.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape2, indata[1].second)) <<
		testutils::sprintf("expected %p, got %p", &shape2, &indata[1].second);
	EXPECT_EQ(src1.type_, type);
	EXPECT_EQ(src1.type_, otype);
	EXPECT_EQ(2, otypes.size());
	EXPECT_EQ(src1.type_, otypes[0]);
	EXPECT_EQ(src2.type_, otypes[1]);

	// original clear validation
	EXPECT_TRUE(pi.input_.data_.expired()) << 
		"moving portal weak pointer did not expire original pointer";
	EXPECT_EQ(nnet::BAD_T, pi.input_.type_);
	EXPECT_FALSE(pi.input_.shape_.is_part_defined()) <<
		"moved portal shape did not make original shape undefined";

	EXPECT_THROW(ai.get_data(ptr, type, shape1), std::exception);

	EXPECT_THROW(oi.get_data(ptr, type, outshape), std::exception);

	// clear
	otype = nnet::BAD_T;
	outdata = nnet::VARR_T{};
	indata.clear();
	otypes.clear();

	passign = std::move(pimv);
	aassign = std::move(aimv);
	oassign = std::move(oimv);

	uuid = std::string((char*) passign.input_.data_.lock().get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, passign.input_.type_);
	EXPECT_TRUE(tensorshape_equal(shape1, passign.input_.shape_)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &passign.input_.shape_);

	ptr = nullptr;
	type = nnet::BAD_T;
	aassign.get_data(ptr, type, shape1);
	uuid = std::string((char*) ptr.get(), src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src1.type_, type);
	
	ptr = nullptr;
	otype = nnet::BAD_T;
	oassign.get_data(ptr, otype, outshape);
	EXPECT_EQ(outdata.first, ptr.get());
	EXPECT_TRUE(tensorshape_equal(outshape, outdata.second)) <<
		testutils::sprintf("expected %p, got %p", &outshape, &outdata.second);
	ASSERT_EQ(2, indata.size());
	uuid = std::string((char*) indata[0].first, src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape1, indata[0].second)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &indata[0].second);
	uuid = std::string((char*) indata[1].first, src2.uuid_.size());
	EXPECT_STREQ(src2.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape2, indata[1].second)) <<
		testutils::sprintf("expected %p, got %p", &shape2, &indata[1].second);
	EXPECT_EQ(src1.type_, otype);
	EXPECT_EQ(2, otypes.size());
	EXPECT_EQ(src1.type_, otypes[0]);
	EXPECT_EQ(src2.type_, otypes[1]);

	// original clear validation
	EXPECT_TRUE(pimv.input_.data_.expired()) <<
		"moving portal weak pointer did not expire original pointer";
	EXPECT_EQ(nnet::BAD_T, pimv.input_.type_);
	EXPECT_FALSE(pimv.input_.shape_.is_part_defined()) <<
		"moved portal shape did not make original shape undefined";

	EXPECT_THROW(aimv.get_data(ptr, type, shape1), std::exception);

	EXPECT_THROW(oimv.get_data(ptr, type, outshape), std::exception);
}


TEST_F(DATA_IO, Portal_E001)
{
	nnet::portal_dest portal;

	nnet::tensorshape shape(random_def_shape(this));
	mock_data_src src(this);
	nnet::tensor arg(shape);
	ASSERT_TRUE(arg.read_from(src)) <<
		"arg failed to read from src";
	ASSERT_TRUE(arg.has_data()) <<
		"arg failed to read from src";
	arg.write_to(portal);

	std::string uuid((char*) portal.input_.data_.lock().get(), src.uuid_.size());
	EXPECT_STREQ(src.uuid_.c_str(), uuid.c_str());
	EXPECT_EQ(src.type_, portal.input_.type_);
	EXPECT_TRUE(tensorshape_equal(shape, portal.input_.shape_)) <<
		testutils::sprintf("expected %p, got %p", &shape, &portal.input_.shape_);
}


TEST_F(DATA_IO, Assign_E002)
{
	nnet::assign_io assign;

	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape(clist);
	nnet::tensorshape bad = make_incompatible(clist);
	mock_data_src src(this);
	{
		nnet::tensor arg(shape);
		ASSERT_TRUE(arg.read_from(src)) <<
			"arg failed to read from src";
		ASSERT_TRUE(arg.has_data()) <<
			"arg failed to read from src";

		std::shared_ptr<void> ptr = nullptr;
		TENS_TYPE type = nnet::BAD_T;
		EXPECT_THROW(assign.get_data(ptr, type, shape), std::exception);

		arg.write_to(assign);
		ptr = nullptr;
		type = nnet::BAD_T;
		assign.get_data(ptr, type, shape);
		std::string uuid((char*) ptr.get(), src.uuid_.size());
		EXPECT_STREQ(src.uuid_.c_str(), uuid.c_str());
		EXPECT_EQ(src.type_, type);
		EXPECT_THROW(assign.get_data(ptr, type, bad), std::exception);
	}
	std::shared_ptr<void> ptr = nullptr;
	TENS_TYPE type = nnet::BAD_T;
	EXPECT_THROW(assign.get_data(ptr, type, shape), std::exception);
}


TEST_F(DATA_IO, Operate_E003)
{
	nnet::tensorshape shape1(random_def_shape(this));
	nnet::tensorshape shape2(random_def_shape(this));
	nnet::tensorshape outshape(random_def_shape(this));

	TENS_TYPE otype = nnet::BAD_T;
	std::vector<TENS_TYPE> otypes;
	nnet::VARR_T outdata;
	std::vector<nnet::CVAR_T> indata;
	nnet::operate_io op(
	[&otype, &outdata, &indata](TENS_TYPE type, nnet::VARR_T dest, std::vector<nnet::CVAR_T> srcs)
	{
		otype = type;
		outdata = dest;
		indata = srcs;
	},
	[&otypes](std::vector<TENS_TYPE> types)
	{
		if (types.empty()) return nnet::BAD_T;
		otypes = types;
		return types[0];
	});

	mock_data_src src1(this);
	mock_data_src src2(this);
	nnet::tensor arg1(shape1);
	nnet::tensor arg2(shape2);
	ASSERT_TRUE(arg1.read_from(src1)) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.read_from(src2)) <<
		"arg2 failed to read from src2";
	ASSERT_TRUE(arg1.has_data()) <<
		"arg1 failed to read from src1";
	ASSERT_TRUE(arg2.has_data()) <<
		"arg2 failed to read from src2";

	std::shared_ptr<void> ptr = nullptr;
	TENS_TYPE type = nnet::BAD_T;
	EXPECT_THROW(op.get_data(ptr, type, shape1), std::exception);

	arg2.write_to(op, 1);
	arg1.write_to(op, 0);

	op.get_data(ptr, type, outshape);
	EXPECT_EQ(outdata.first, ptr.get());
	EXPECT_TRUE(tensorshape_equal(outshape, outdata.second)) <<
		testutils::sprintf("expected %p, got %p", &outshape, &outdata.second);
	ASSERT_EQ(2, indata.size());
	std::string uuid((char*) indata[0].first, src1.uuid_.size());
	EXPECT_STREQ(src1.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape1, indata[0].second)) <<
		testutils::sprintf("expected %p, got %p", &shape1, &indata[0].second);
	uuid = std::string((char*) indata[1].first, src2.uuid_.size());
	EXPECT_STREQ(src2.uuid_.c_str(), uuid.c_str());
	EXPECT_TRUE(tensorshape_equal(shape2, indata[1].second)) <<
		testutils::sprintf("expected %p, got %p", &shape2, &indata[1].second);
	EXPECT_EQ(src1.type_, type);
	EXPECT_EQ(src1.type_, otype);
	EXPECT_EQ(2, otypes.size());
	EXPECT_EQ(src1.type_, otypes[0]);
	EXPECT_EQ(src2.type_, otypes[1]);
}


#endif /* DISABLE_DIO_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */
