
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"

#include "common.hpp"


struct FUNCTOR : public ::testing::Test {};


TEST_F(FUNCTOR, Shapes)
{
	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	std::vector<ade::DimT> bad = {94, 78, 70, 82, 62, 22, 38};
	ade::Shape shape(slist);
	ade::Shape badshape(bad);

	ade::Tensorptr leaf = new MockTensor(shape);
	ade::Tensorptr leaf1 = new MockTensor(shape);
	ade::Tensorptr badleaf = new MockTensor(badshape);

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ade::Shape gotshape = func->shape();
	EXPECT_ARREQ(shape, gotshape);

	EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0}, {}),
		"cannot perform MOCK with no arguments");

	// EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0},
	// 	{{ade::identity, leaf}, {ade::identity, nullptr}}),
	// 	"cannot perform MOCK with null arguments");

	std::string fatalmsg = err::sprintf("cannot perform MOCK with incompatible shapes %s and %s",
		shape.to_string().c_str(), badshape.to_string().c_str());
	EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		{ade::identity, leaf}, {ade::identity, badleaf}}), fatalmsg.c_str());
}


TEST_F(FUNCTOR, Opcode)
{
	std::string mockname = "asd123101ksq";
	size_t mockcode = 3247;
	ade::Tensorptr leaf = new MockTensor();

	ade::Functor* func = ade::Functor::get(ade::Opcode{mockname, mockcode},
		{{ade::identity, leaf}});

	ade::Opcode op = func->get_opcode();
	EXPECT_STREQ(mockname.c_str(), op.name_.c_str());
	EXPECT_EQ(mockcode, op.code_);

	delete func;
}


TEST_F(FUNCTOR, Childrens)
{
	ade::Tensorptr leaf = new MockTensor();
	ade::Tensorptr leaf1 = new MockTensor();

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	ade::ArgsT refs = static_cast<ade::iFunctor*>(func.get())->get_children();

	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0].tensor_.get());
	EXPECT_EQ(leaf1.get(), refs[1].tensor_.get());
}


TEST_F(FUNCTOR, ToString)
{
	ade::Tensorptr leaf = new MockTensor();
	ade::Tensorptr leaf1 = new MockTensor();

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
