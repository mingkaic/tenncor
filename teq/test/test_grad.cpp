
#ifndef DISABLE_GRAD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "teq/grad_def.hpp"


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
			auto ochild = static_cast<const MockEdge*>(&op->get_children()[arg_idx].get());
			return std::make_shared<MockFunctor>(
				MockEdgesT{MockEdge(*ochild)}, teq::Opcode{"FUNC4", 3});
		}
		return teq::TensptrT(new MockTensor(op->shape(), "other"));
	}

	teq::TensptrT chain_rule (teq::FuncptrT op, const teq::TensptrT& local_der,
		teq::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		teq::TensptrT tens(new MockFunctor(
			teq::TensptrsT{op,local_der}, teq::Opcode{"FUNC2", 1}));

		return teq::TensptrT(new MockFunctor(
			teq::TensptrsT{tens, supcomp_grad}, teq::Opcode{"FUNC3", 2}));
	}

	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockTensor(shape, "1"));
	}

	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockTensor(shape, "0"));
	}

	teq::TensptrT add (teq::TensptrT& lhs, teq::TensptrT& rhs) const override
	{
		return teq::TensptrT(new MockFunctor(
			teq::TensptrsT{lhs, rhs}, teq::Opcode{"FUNC", 0}));
	}
};


TEST(GRAD, OneZero)
{
	MockGradientBuilder builder;

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new MockTensor(shape, "leaf"));
	teq::TensptrT leaf1(new MockTensor(shape, "leaf2"));
	teq::TensptrT leaf2(new MockTensor(shape, "leaf3"));
	teq::TensptrT f(new MockFunctor(
		teq::TensptrsT{leaf, leaf1}, teq::Opcode{"FUNC", 0}));

	auto wun = builder.derive(f, f);
	auto wun2 = builder.derive(leaf, leaf);
	auto wun3 = builder.derive(leaf2, leaf2);

	std::string shapestr = shape.to_string();
	EXPECT_STREQ("1", wun->to_string().c_str());
	EXPECT_STREQ("1", wun2->to_string().c_str());
	EXPECT_STREQ("1", wun3->to_string().c_str());

	auto zro = builder.derive(leaf, leaf2);
	auto zro2 = builder.derive(leaf2, leaf);
	auto zro3 = builder.derive(f, leaf2);
	auto zro4 = builder.derive(leaf, nullptr);
	auto zro5 = builder.derive(nullptr, leaf);

	EXPECT_STREQ("0", zro->to_string().c_str());
	EXPECT_STREQ("0", zro2->to_string().c_str());
	EXPECT_STREQ("0", zro3->to_string().c_str());
	EXPECT_STREQ("0", zro4->to_string().c_str());
	EXPECT_STREQ("0", zro5->to_string().c_str());
}


TEST(GRAD, BuilderStandardV)
{
	MockGradientBuilder builder;

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT leaf1(new LabelledMockTensor("leaf2", shape));
	teq::TensptrT f(new MockFunctor(
		teq::TensptrsT{leaf, leaf1}, teq::Opcode{"FUNC", 0}));

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

	/* diamond
	 *
	 *   leaf
	 *   /   \
	 * FUNC  FUNC2
	 *   \   /
	 *   FUNC3
	 */
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT f(new MockFunctor(
		teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(
		teq::TensptrsT{leaf}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(
		teq::TensptrsT{f, f2}, teq::Opcode{"FUNC3", 2}));

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

	/* diamond with a tail
	 *
	 *   leaf
	 *    |
	 *   FUNC
	 *   /   \
	 * FUNC2  FUNC3
	 *   \   /
	 *   FUNC4
	 */
	teq::TensptrT leaf(new LabelledMockTensor("leaf", shape));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{f}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f}, teq::Opcode{"FUNC3", 2}));
	teq::TensptrT f4(new MockFunctor(teq::TensptrsT{f2, f3}, teq::Opcode{"FUNC4", 3}));

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
