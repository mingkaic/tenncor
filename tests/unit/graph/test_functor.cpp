//
// Created by Mingkai Chen on 2017-03-14.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include "gtest/gtest.h"

#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"
#include "mock_node.hpp"
#include "mock_observer.hpp"

#include "graph/functor.hpp"
#include "graph/variable.hpp"


#ifndef DISABLE_FUNCTOR_TEST


class FUNCTOR : public testify::fuzz_test
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


nnet::tensor* trash_fwd (std::unique_ptr<nnet::idata_src>&, std::vector<nnet::inode*>)
{
	return nullptr;
}


nnet::varptr trash_bwd (nnet::inode*, std::vector<nnet::inode*>)
{
	return nullptr;
}


// covers functor: clone
TEST_F(FUNCTOR, Copy_F000)
{
	mock_node badleaf("badleef");
	nnet::functor* assign = nnet::functor::get({&badleaf}, trash_fwd, trash_bwd, (OPCODE) 0);
	nnet::functor* assign2 = nnet::functor::get({&badleaf}, trash_fwd, trash_bwd, (OPCODE) 0);
	mock_observer* obs = new mock_observer({assign});

	mock_node goodleaf("gooflead");
	double c = get_double(1, "c")[0];
	double c2 = get_double(1, "c2")[0];
	nnet::constant* arg = nnet::constant::get<double>(c);
	nnet::functor* func = nnet::functor::get({arg}, 
		[c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(std::vector<size_t>{1});
		}, 
		[&goodleaf](nnet::inode*, std::vector<nnet::inode*>)
		{
			return &goodleaf;
		}, (OPCODE) 0);
	
	size_t counter = 0;
	std::shared_ptr<nnet::const_init> ci = std::make_shared<nnet::const_init>();
	ci->set<double>(c);
	nnet::variable var(nnet::tensorshape(std::vector<size_t>{1}), ci, "variable");
	nnet::functor* func2 = nnet::functor::get({&var},
		[&counter, c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			counter++;
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(std::vector<size_t>{1});
		}, trash_bwd, (OPCODE) 0);

	nnet::tensor* ten = func->get_tensor(); // this belongs to func
	ASSERT_EQ(&goodleaf, func->derive(&badleaf).get());
	ASSERT_EQ(c2, nnet::expose<double>(ten)[0]);
	EXPECT_EQ(0, counter);

	nnet::functor* cp = func->clone();
	nnet::functor* cp2 = func2->clone();
	// copy over tensor
	EXPECT_NE(ten, cp->get_tensor());
	std::vector<double> cpvec = nnet::expose<double>(cp);
	EXPECT_EQ(1, cpvec.size());
	EXPECT_EQ(c2, cpvec[0]);

	// test backward copy over
	nnet::varptr cpgrad = cp->derive(&badleaf);
	EXPECT_EQ(&goodleaf, cpgrad.get());

	*assign = *cp;
	*assign2 = *cp2;

	// copy over tensor
	EXPECT_NE(ten, assign->get_tensor());
	std::vector<double> asvec = nnet::expose<double>(cp);
	EXPECT_EQ(1, asvec.size());
	EXPECT_EQ(c2, asvec[0]);

	// test backward copy over
	nnet::varptr asgrad = assign->derive(&badleaf);
	EXPECT_EQ(&goodleaf, asgrad.get());

	// test notification
	size_t n_updates = testify::mocker::get_usage(obs, "update2");
	optional<std::string> updateval = testify::mocker::get_value(obs, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"obs update2 value is not found";;
	EXPECT_STREQ("UPDATE", updateval->c_str());

	// test forward copy over
	var.initialize(); // increment both assign and cp
	ASSERT_EQ(3, counter);
	
	delete arg;
}


// covers functor: move
TEST_F(FUNCTOR, Move_F000)
{
	mock_node badleaf("badleef");
	nnet::functor* assign = nnet::functor::get({&badleaf}, trash_fwd, trash_bwd, (OPCODE) 0);
	nnet::functor* assign2 = nnet::functor::get({&badleaf}, trash_fwd, trash_bwd, (OPCODE) 0);
	mock_observer* obs = new mock_observer({assign});

	mock_node goodleaf("gooflead");
	double c = get_double(1, "c")[0];
	double c2 = get_double(1, "c2")[0];
	nnet::constant* arg = nnet::constant::get<double>(c);
	nnet::functor* func = nnet::functor::get({arg}, 
		[c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(std::vector<size_t>{1});
		},
		[&goodleaf](nnet::inode*, std::vector<nnet::inode*>)
		{
			return &goodleaf;
		}, (OPCODE) 0);
	
	size_t counter = 0;
	std::shared_ptr<nnet::const_init> ci = std::make_shared<nnet::const_init>();
	ci->set<double>(c);
	nnet::variable var(nnet::tensorshape(std::vector<size_t>{1}), ci, "variable");
	nnet::functor* func2 = nnet::functor::get({&var},
		[&counter, c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			counter++;
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(std::vector<size_t>{1});
		}, trash_bwd, (OPCODE) 0);

	nnet::tensor* ten = func->get_tensor(); // this belongs to func
	ASSERT_EQ(&goodleaf, func->derive(&badleaf).get());
	ASSERT_EQ(c2, nnet::expose<double>(ten)[0]);
	EXPECT_EQ(0, counter);

	nnet::functor* mv = func->move();
	nnet::functor* mv2 = func2->move();
	// move over tensor
	EXPECT_EQ(ten, mv->get_tensor());
	EXPECT_EQ(nullptr, func->get_tensor());

	// test backward move over
	nnet::varptr mvgrad = mv->derive(&badleaf);
	EXPECT_EQ(&goodleaf, mvgrad.get());

	*assign = std::move(*mv);
	*assign2 = std::move(*mv2);

	// move over tensor
	EXPECT_EQ(ten, assign->get_tensor());
	EXPECT_EQ(nullptr, mv->get_tensor());

	// test backward move over
	nnet::varptr asgrad = assign->derive(&badleaf);
	EXPECT_EQ(&goodleaf, asgrad.get());

	// test notification
	size_t n_updates = testify::mocker::get_usage(obs, "update2");
	optional<std::string> updateval = testify::mocker::get_value(obs, "update2");
	EXPECT_EQ(1, n_updates);
	ASSERT_TRUE((bool) updateval) <<
		"obs update2 value is not found";;
	EXPECT_STREQ("UPDATE", updateval->c_str());

	// test forward move over
	var.initialize(); // increment both assign and mv
	ASSERT_EQ(1, counter);

	delete arg;
}


// covers functor: get_name
TEST_F(FUNCTOR, Name_F001)
{
	std::string leaflabel = get_string(get_int(1, "leaflabel.size", {14, 29})[0], "leaflabel");
	mock_node leaf(leaflabel);
	nnet::functor* func = nnet::functor::get({&leaf}, trash_fwd, trash_bwd, (OPCODE) 0);

	std::string expectname = "<ABS:" + func->get_uid() + ">(" + leaflabel + ")";
	EXPECT_STREQ(expectname.c_str(), func->get_name().c_str());
}


// covers functor: get_leaves
TEST_F(FUNCTOR, GetLeaves_F002)
{
	std::string leaflabel = get_string(get_int(1, "leaflabel.size", {14, 29})[0], "leaflabel");
	std::string leaflabel2 = get_string(get_int(1, "leaflabel.size", {14, 29})[0], "leaflabel2");
	mock_node leaf(leaflabel);
	mock_node leaf2(leaflabel2);
	nnet::functor* func = nnet::functor::get({&leaf}, trash_fwd, trash_bwd, (OPCODE) 0);
	nnet::functor* func2 = nnet::functor::get({func, &leaf2}, trash_fwd, trash_bwd, (OPCODE) 0);

	std::unordered_set<const nnet::inode*> leafset = func->get_leaves();
	std::unordered_set<const nnet::inode*> leafset2 = func2->get_leaves();
	ASSERT_EQ(1, leafset.size());
	EXPECT_EQ(&leaf, *(leafset.begin())) << "leaf variable not found in leafset";
	ASSERT_EQ(2, leafset2.size());
	EXPECT_TRUE(leafset2.end() != leafset2.find(&leaf)) << 
		"leaf variable not found in leafset2";
	EXPECT_TRUE(leafset2.end() != leafset2.find(&leaf2)) << 
		"leaf2 variable not found in leafset2";
}


// covers variable: get_tensor
TEST_F(FUNCTOR, GetTensor_F003)
{
	std::string varlabel = get_string(get_int(1, "varlabel.size", {14, 29})[0], "varlabel");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensorshape shape2 = random_def_shape(this);
	double c = get_double(1, "c")[0];
	double c2 = get_double(1, "c2")[0];
	std::shared_ptr<nnet::const_init> ci = std::make_shared<nnet::const_init>();
	ci->set<double>(c);
	nnet::variable res(shape, ci, varlabel);
	nnet::functor* func = nnet::functor::get({&res}, 
		[shape2, c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(shape2);
		}, trash_bwd, (OPCODE) 0);

	EXPECT_EQ(nullptr, func->get_tensor());
	res.initialize();
	nnet::tensor* ten = func->get_tensor();
	EXPECT_TRUE(ten->has_data()) <<
		"functor tensor does not have data";
	nnet::tensorshape gotshape = ten->get_shape();
	ASSERT_TRUE(tensorshape_equal(shape2, gotshape)) <<
		sprintf("expecting shape %p, got %p", &shape2, &gotshape);
	std::vector<double> dvec = nnet::expose<double>(func);
	EXPECT_EQ(gotshape.n_elems(), dvec.size());
	for (double d : dvec)
	{
		ASSERT_EQ(c2, d);
	}
}


// covers variable: derive
TEST_F(FUNCTOR, Derive_F004)
{
	std::string varlabel = get_string(get_int(1, "varlabel.size", {14, 29})[0], "varlabel");
	nnet::tensorshape shape = random_def_shape(this);
	nnet::tensorshape shape2 = random_def_shape(this);
	double c = get_double(1, "c")[0];
	double c2 = get_double(1, "c2")[0];
	mock_node goodleaf("gooflead");

	nnet::constant* arg = nnet::constant::get<double>(c);
	nnet::functor* func = nnet::functor::get({arg}, 
		[shape2, c2](std::unique_ptr<nnet::idata_src>& src, std::vector<nnet::inode*>)
		{
			nnet::const_init* ci = new nnet::const_init;
			ci->set<double>(c2);
			src = std::unique_ptr<nnet::idata_src>(ci);
			return new nnet::tensor(shape2);
		}, 
		[&goodleaf](nnet::inode*, std::vector<nnet::inode*>)
		{
			return &goodleaf;
		}, (OPCODE) 0);

	nnet::varptr eleaf = func->derive(nullptr);
	nnet::varptr ewun = func->derive(func);

	EXPECT_EQ(&goodleaf, eleaf.get());
	nnet::tensor* wten = ewun->get_tensor();
	ASSERT_NE(nullptr, wten);
	nnet::tensorshape gotshape = wten->get_shape();
	ASSERT_TRUE(tensorshape_equal(shape2, gotshape)) <<
		sprintf("expecting shape %p, got %p", &shape2, &gotshape);
	std::vector<double> wunvec = nnet::expose<double>(ewun);
	size_t n = shape2.n_elems();
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(1, wunvec[i]);
	}

	delete arg;
}


#endif /* DISABLE_FUNCTOR_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
