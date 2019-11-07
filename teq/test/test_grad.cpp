
#ifndef DISABLE_GRAD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"

#include "teq/grad_def.hpp"
#include "teq/functor.hpp"


struct LabelledMockTensor final : public MockTensor
{
	LabelledMockTensor (std::string label, teq::Shape shape) :
		MockTensor(shape), label_(label) {}

	std::string to_string (void) const override
	{
		return label_;
	}

	std::string label_;
};


struct MockGradientBuilder final : public teq::iGradientBuilder
{
	teq::TensptrT local_derivative (teq::FuncptrT op, size_t arg_idx) const override
	{
		std::string label = op->to_string();
		if (label == "FUNC")
		{
			return op->get_children()[arg_idx].get().get_tensor();
		}
		else if (label == "FUNC2")
		{
			return teq::TensptrT(teq::Functor::get(teq::Opcode{"FUNC4", 3},
				{*static_cast<const teq::FuncArg*>(&op->get_children()[arg_idx].get())}));
		}
		return teq::TensptrT(new LabelledMockTensor("other", op->shape()));
	}

	teq::TensptrT chain_rule (teq::FuncptrT op, const teq::TensptrT& local_der,
		teq::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		teq::TensptrT tens(teq::Functor::get(teq::Opcode{"FUNC2", 1}, {
			teq::identity_map(op),
			teq::identity_map(local_der),
		}));

		return teq::TensptrT(teq::Functor::get(teq::Opcode{"FUNC3", 2}, {
			teq::identity_map(tens),
			teq::identity_map(supcomp_grad),
		}));
	}

	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return teq::TensptrT(new LabelledMockTensor("1", shape));
	}

	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return teq::TensptrT(new LabelledMockTensor("0", shape));
	}

	teq::TensptrT add (teq::TensptrT& lhs, teq::TensptrT& rhs) const override
	{
		return teq::TensptrT(teq::Functor::get(teq::Opcode{"FUNC", 0}, {
			teq::identity_map(lhs),
			teq::identity_map(rhs),
		}));
	}
};


TEST(GRAD, OneZero)
{
	MockGradientBuilder builder;

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	// standard v
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT leaf1(new LabelledMockTensor("leaf2", shape));
	teq::TensptrT leaf2(new LabelledMockTensor("leaf3", shape));
	teq::TensptrT f(teq::Functor::get(teq::Opcode{"FUNC", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(leaf1),
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

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	// standard v
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT leaf1(new LabelledMockTensor("leaf2", shape));
	teq::TensptrT f(teq::Functor::get(teq::Opcode{"FUNC", 0}, {
		teq::identity_map(leaf),
		teq::identity_map(leaf1),
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

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	// diamond
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT f(teq::Functor::get(teq::Opcode{"FUNC", 0}, {
		teq::identity_map(leaf),
	}));
	teq::TensptrT f2(teq::Functor::get(teq::Opcode{"FUNC2", 1}, {
		teq::identity_map(leaf),
	}));
	teq::TensptrT f3(teq::Functor::get(teq::Opcode{"FUNC3", 2}, {
		teq::identity_map(f),
		teq::identity_map(f2),
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

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT f(teq::Functor::get(teq::Opcode{"FUNC", 0}, {
		teq::identity_map(leaf),
	}));
	teq::TensptrT f2(teq::Functor::get(teq::Opcode{"FUNC2", 1}, {
		teq::identity_map(f),
	}));
	teq::TensptrT f3(teq::Functor::get(teq::Opcode{"FUNC3", 2}, {
		teq::identity_map(f),
	}));
	teq::TensptrT f4(teq::Functor::get(teq::Opcode{"FUNC4", 3}, {
		teq::identity_map(f2),
		teq::identity_map(f3),
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
