
#ifndef DISABLE_GRAD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "teq/derive.hpp"


struct MockDeriveFunc final : public teq::iDerivativeFuncs
{
	teq::TensptrT local_derivative (teq::FuncptrT op, size_t arg_idx) const override
	{
		std::string label = op->to_string();
		if (label == "FUNC")
		{
			return op->get_children()[arg_idx];
		}
		else if (label == "FUNC2")
		{
			return std::make_shared<MockFunctor>(
				teq::TensptrsT{op->get_children()[arg_idx]},
				std::vector<double>{}, teq::Opcode{"FUNC4", 3});
		}
		return teq::TensptrT(new MockLeaf({}, op->shape(), "other"));
	}

	teq::TensptrT chain_rule (teq::FuncptrT op, const teq::TensptrT& local_der,
		teq::TensptrT supcomp_grad, size_t arg_idx) const override
	{
		teq::TensptrT tens(new MockFunctor(
			teq::TensptrsT{op,local_der},
			std::vector<double>{}, teq::Opcode{"FUNC2", 1}));

		return teq::TensptrT(new MockFunctor(
			teq::TensptrsT{tens, supcomp_grad},
			std::vector<double>{}, teq::Opcode{"FUNC3", 2}));
	}

	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockLeaf({}, shape, "1"));
	}

	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockLeaf({}, shape, "0"));
	}

	teq::TensptrT add (teq::TensptrsT elems) const override
	{
		return teq::TensptrT(new MockFunctor(elems,
			std::vector<double>{}, teq::Opcode{"FUNC", 0}));
	}
};


TEST(GRAD, OneZero)
{
	MockDeriveFunc builder;

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new MockLeaf({}, shape, "leaf"));
	teq::TensptrT leaf2(new MockLeaf({}, shape, "leaf2"));
	teq::TensptrT leaf3(new MockLeaf({}, shape, "leaf3"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf, leaf2},
		std::vector<double>{}, teq::Opcode{"FUNC", 0}));

	auto wun = teq::derive(f, f, builder);
	auto wun2 = teq::derive(leaf, leaf, builder);
	auto wun3 = teq::derive(leaf3, leaf3, builder);

	std::string shapestr = shape.to_string();
	EXPECT_STREQ("1", wun->to_string().c_str());
	EXPECT_STREQ("1", wun2->to_string().c_str());
	EXPECT_STREQ("1", wun3->to_string().c_str());

	auto zro = teq::derive(leaf, leaf3, builder);
	auto zro2 = teq::derive(leaf3, leaf, builder);
	auto zro3 = teq::derive(f, leaf3, builder);
	auto zro4 = teq::derive(leaf, nullptr, builder);
	auto zro5 = teq::derive(nullptr, leaf, builder);

	EXPECT_STREQ("0", zro->to_string().c_str());
	EXPECT_STREQ("0", zro2->to_string().c_str());
	EXPECT_STREQ("0", zro3->to_string().c_str());
	EXPECT_STREQ("0", zro4->to_string().c_str());
	EXPECT_STREQ("0", zro5->to_string().c_str());
}


TEST(GRAD, BuilderStandardV)
{
	MockDeriveFunc builder;

	std::vector<teq::DimT> slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new MockLeaf({}, shape, "leaf"));
	teq::TensptrT leaf2(new MockLeaf({}, shape, "leaf2"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf, leaf2},
		std::vector<double>{}, teq::Opcode{"FUNC", 0}));

	auto gl = teq::derive(f, leaf, builder);
	auto gl2 = teq::derive(f, leaf2, builder);

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
	MockDeriveFunc builder;

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
	teq::TensptrT leaf(new MockLeaf({}, shape, "leaf"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf},
		std::vector<double>{}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{leaf},
		std::vector<double>{}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f, f2},
		std::vector<double>{}, teq::Opcode{"FUNC3", 2}));

	auto gl = teq::derive(f3, leaf, builder);

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
	MockDeriveFunc builder;

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
	teq::TensptrT leaf(new MockLeaf({}, shape, "leaf"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf},
		std::vector<double>{}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{f},
		std::vector<double>{}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f},
		std::vector<double>{}, teq::Opcode{"FUNC3", 2}));
	teq::TensptrT f4(new MockFunctor(teq::TensptrsT{f2, f3},
		std::vector<double>{}, teq::Opcode{"FUNC4", 3}));

	auto gl = teq::derive(f4, leaf, builder);
	EXPECT_GRAPHEQ(
		"(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" |   `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		" `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     |       `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"     `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"         `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   `--(FUNC4[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |   `--(FUNC2[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |   |   `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |   |       `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |   `--(FUNC3[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |       `--(FUNC[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   |           `--(constant:leaf[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             |   `--(constant:other[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"             `--(constant:1[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);
}


#endif // DISABLE_GRAD_TEST
