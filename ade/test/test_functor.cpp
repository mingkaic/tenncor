
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

	ade::TensptrT leaf(new MockTensor(shape));
	ade::TensptrT leaf1(new MockTensor(shape));
	ade::TensptrT badleaf(new MockTensor(badshape));

	ade::TensptrT func(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ade::Shape gotshape = func->shape();
	EXPECT_ARREQ(shape, gotshape);

	EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0}, {}),
		"cannot perform MOCK with no arguments");

	std::string fatalmsg = fmts::sprintf(
		"cannot perform MOCK with incompatible shapes %s and %s",
		shape.to_string().c_str(), badshape.to_string().c_str());
	EXPECT_FATAL(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(badleaf),
	}), fatalmsg.c_str());
}


TEST_F(FUNCTOR, Opcode)
{
	std::string mockname = "asd123101ksq";
	size_t mockcode = 3247;
	ade::TensptrT leaf(new MockTensor());

	ade::Functor* func = ade::Functor::get(ade::Opcode{mockname, mockcode}, {
		ade::identity_map(leaf),
	});

	ade::Opcode op = func->get_opcode();
	EXPECT_STREQ(mockname.c_str(), op.name_.c_str());
	EXPECT_EQ(mockcode, op.code_);

	delete func;
}


TEST_F(FUNCTOR, Childrens)
{
	ade::TensptrT leaf(new MockTensor());
	ade::TensptrT leaf1(new MockTensor());

	ade::TensptrT func(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ASSERT_NE(nullptr, func.get());

	ade::ArgsT refs = static_cast<ade::iFunctor*>(func.get())->get_children();

	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0].get_tensor().get());
	EXPECT_EQ(leaf1.get(), refs[1].get_tensor().get());
}


TEST_F(FUNCTOR, ToString)
{
	ade::TensptrT leaf(new MockTensor());
	ade::TensptrT leaf1(new MockTensor());

	ade::TensptrT func(ade::Functor::get(ade::Opcode{"MOCK", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
