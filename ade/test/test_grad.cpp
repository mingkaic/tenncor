
#ifndef DISABLE_GRAD_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "ade/test/common.hpp"

#include "ade/grad_def.hpp"
#include "ade/functor.hpp"

#include "testutil/tutil.hpp"


struct LabelledMockTensor final : public MockTensor
{
	LabelledMockTensor (std::string label, ade::Shape shape) :
		MockTensor(shape), label_(label) {}

	std::string to_string (void) const override
	{
		return label_;
	}

	std::string label_;
};


struct MockGradientBuilder final : public ade::iGradientBuilder
{
	ade::TensptrT local_derivative (ade::FuncptrT op, size_t arg_idx) const override
	{
		std::string label = op->to_string();
		if (label == "FUNC")
		{
			return op->get_children()[arg_idx].get_tensor();
		}
		else if (label == "FUNC2")
		{
			return ade::TensptrT(ade::Functor::get(ade::Opcode{"FUNC4", 3},
				{op->get_children()[arg_idx]}));
		}
		return ade::TensptrT(new LabelledMockTensor("other", op->shape()));
	}

	ade::TensptrT chain_rule (ade::FuncptrT op, const ade::TensptrT& local_der,
		ade::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		ade::TensptrT tens(ade::Functor::get(ade::Opcode{"FUNC2", 1}, {
			ade::identity_map(op),
			ade::identity_map(local_der),
		}));

		return ade::TensptrT(ade::Functor::get(ade::Opcode{"FUNC3", 2}, {
			ade::identity_map(tens),
			ade::identity_map(supcomp_grad),
		}));
	}

	ade::TensptrT get_const_one (ade::Shape shape) const override
	{
		return ade::TensptrT(new LabelledMockTensor("1", shape));
	}

	ade::TensptrT get_const_zero (ade::Shape shape) const override
	{
		return ade::TensptrT(new LabelledMockTensor("0", shape));
	}

	ade::TensptrT add (ade::TensptrT& lhs, ade::TensptrT& rhs) const override
	{
		return ade::TensptrT(ade::Functor::get(ade::Opcode{"FUNC", 0}, {
			ade::identity_map(lhs),
			ade::identity_map(rhs),
		}));
	}
};


TEST(GRAD, OneZero)
{
	MockGradientBuilder builder;

	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Shape shape(slist);

	// standard v
	ade::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	ade::TensptrT leaf1(new LabelledMockTensor("leaf2", shape));
	ade::TensptrT leaf2(new LabelledMockTensor("leaf3", shape));
	ade::TensptrT f(ade::Functor::get(ade::Opcode{"FUNC", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	auto wun = builder.derive(f, f);
	auto wun2 = builder.derive(leaf, leaf);
	auto wun3 = builder.derive(leaf2, leaf2);

	EXPECT_STREQ("1", wun->to_string().c_str());
	EXPECT_STREQ("1", wun2->to_string().c_str());
	EXPECT_STREQ("1", wun3->to_string().c_str());

	auto zro = builder.derive(leaf, leaf2);
	auto zro2 = builder.derive(leaf2, leaf);
	auto zro3 = builder.derive(f, leaf2);

	EXPECT_STREQ("0", zro->to_string().c_str());
	EXPECT_STREQ("0", zro2->to_string().c_str());
	EXPECT_STREQ("0", zro3->to_string().c_str());
}


TEST(GRAD, BuilderStandardV)
{
	MockGradientBuilder builder;

	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Shape shape(slist);

	// standard v
	ade::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	ade::TensptrT leaf1(new LabelledMockTensor("leaf2", shape));
	ade::TensptrT f(ade::Functor::get(ade::Opcode{"FUNC", 0}, {
		ade::identity_map(leaf),
		ade::identity_map(leaf1),
	}));

	auto gl = builder.derive(f, leaf);
	auto gl2 = builder.derive(f, leaf1);

	EXPECT_GRAPHEQ(
		"(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);

	EXPECT_GRAPHEQ(
		"(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(constant:leaf2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])",
		gl2);
}


TEST(GRAD, BuilderDiamond)
{
	MockGradientBuilder builder;

	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Shape shape(slist);

	// diamond
	ade::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	ade::TensptrT f(ade::Functor::get(ade::Opcode{"FUNC", 0}, {
		ade::identity_map(leaf),
	}));
	ade::TensptrT f2(ade::Functor::get(ade::Opcode{"FUNC2", 1}, {
		ade::identity_map(leaf),
	}));
	ade::TensptrT f3(ade::Functor::get(ade::Opcode{"FUNC3", 2}, {
		ade::identity_map(f),
		ade::identity_map(f2),
	}));

	auto gl = builder.derive(f3, leaf);

	EXPECT_GRAPHEQ(
		"(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);
}


TEST(GRAD, TadPole)
{
	MockGradientBuilder builder;

	std::vector<ade::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	ade::Shape shape(slist);

	ade::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	ade::TensptrT f(ade::Functor::get(ade::Opcode{"FUNC", 0}, {
		ade::identity_map(leaf),
	}));
	ade::TensptrT f2(ade::Functor::get(ade::Opcode{"FUNC2", 1}, {
		ade::identity_map(f),
	}));
	ade::TensptrT f3(ade::Functor::get(ade::Opcode{"FUNC3", 2}, {
		ade::identity_map(f),
	}));
	ade::TensptrT f4(ade::Functor::get(ade::Opcode{"FUNC4", 3}, {
		ade::identity_map(f2),
		ade::identity_map(f3),
	}));

	auto gl = builder.derive(f4, leaf);
	EXPECT_GRAPHEQ(
		"(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);
}


#endif // DISABLE_GRAD_TEST
