
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "ade/functor.hpp"

#include "testutil/common.hpp"

#include "common.hpp"


struct FUNCTOR : public simple::TestModel {};


TEST_F(FUNCTOR, Shapes)
{
	simple::SessionT sess = get_session("FUNCTOR::Shape");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> bad = get_incompatible(sess, slist, "bad");
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

	std::string fatalmsg = err::sprintf("cannot perform MOCK with incompatible shapes %s and %s",
		shape.to_string().c_str(), badshape.to_string().c_str());
	EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		{ade::identity, leaf}, {ade::identity, badleaf}}), fatalmsg.c_str());
}


TEST_F(FUNCTOR, Opcode)
{
	simple::SessionT sess = get_session("FUNCTOR::Opcode");

	std::string mockname = sess->get_string("mockname", 10);
	size_t mockcode = sess->get_int("mockcode", 1, {0, 1312})[0];
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
	simple::SessionT sess = get_session("FUNCTOR::ToString");

	ade::Tensorptr leaf = new MockTensor();
	ade::Tensorptr leaf1 = new MockTensor();

	ade::Tensorptr func = ade::Functor::get(ade::Opcode{"MOCK", 0},
		{{ade::identity, leaf}, {ade::identity, leaf1}});

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
