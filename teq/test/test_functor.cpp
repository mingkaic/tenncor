
#ifndef DISABLE_FUNCTOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/test/common.hpp"

#include "teq/functor.hpp"


TEST(FUNCTOR, Shapes)
{
	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	std::vector<teq::DimT> bad = {94, 78, 70, 82, 62, 22, 38};
	teq::Shape shape(slist);
	teq::Shape badshape(bad);

	teq::TensptrT leaf(new MockTensor(shape));
	teq::TensptrT leaf1(new MockTensor(shape));
	teq::TensptrT badleaf(new MockTensor(badshape));

	teq::TensptrT func(teq::Functor::get(teq::Opcode{"MOCK", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(leaf1),
	}));

	teq::Shape gotshape = func->shape();
	EXPECT_ARREQ(shape, gotshape);

	EXPECT_FATAL(teq::Functor::get(teq::Opcode{"MOCK", 0}, {}),
		"cannot perform `MOCK` with no arguments");

	std::string fatalmsg = fmts::sprintf(
		"cannot perform `MOCK` with incompatible shapes %s and %s",
		shape.to_string().c_str(), badshape.to_string().c_str());
	EXPECT_FATAL(teq::Functor::get(teq::Opcode{"MOCK", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(badleaf),
	}), fatalmsg.c_str());
}


TEST(FUNCTOR, Opcode)
{
	std::string mockname = "asd123101ksq";
	size_t mockcode = 3247;
	teq::TensptrT leaf(new MockTensor());

	teq::Functor* func = teq::Functor::get(teq::Opcode{mockname, mockcode}, {
		teq::identity_map(leaf),
	});

	teq::Opcode op = func->get_opcode();
	EXPECT_STREQ(mockname.c_str(), op.name_.c_str());
	EXPECT_EQ(mockcode, op.code_);

	delete func;
}


TEST(FUNCTOR, Children)
{
	teq::TensptrT leaf(new MockTensor());
	teq::TensptrT leaf1(new MockTensor());
	teq::TensptrT leaf2(new MockTensor());

	teq::FuncptrT func(teq::Functor::get(teq::Opcode{"MOCK", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(leaf1),
	}));

	ASSERT_NE(nullptr, func.get());

	teq::ArgsT refs = func->get_children();

	ASSERT_EQ(2, refs.size());
	EXPECT_EQ(leaf.get(), refs[0].get_tensor().get());
	EXPECT_EQ(leaf1.get(), refs[1].get_tensor().get());

	EXPECT_WARN((func->update_child(teq::identity_map(leaf2), 1)),
		"teq::Functor does not allow editing of children");
}


TEST(FUNCTOR, ToString)
{
	teq::TensptrT leaf(new MockTensor());
	teq::TensptrT leaf1(new MockTensor());

	teq::TensptrT func(teq::Functor::get(teq::Opcode{"MOCK", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(leaf1),
	}));

	ASSERT_NE(nullptr, func.get());

	EXPECT_STREQ("MOCK", func->to_string().c_str());
}


#endif // DISABLE_FUNCTOR_TEST
