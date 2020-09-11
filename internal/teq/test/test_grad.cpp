
#ifndef DISABLE_TEQ_GRAD_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/leaf.hpp"
#include "internal/teq/mock/functor.hpp"

#include "internal/teq/derive.hpp"


struct MockDeriveFunc final : public teq::iDerivativeFuncs
{
	teq::TensptrT lderive (teq::FuncptrT op,
		teq::TensptrT supgrad, size_t arg_idx) const override
	{
		std::string label = op->to_string();
		teq::TensptrT local_der;
		if (label == "FUNC")
		{
			local_der = op->get_args()[arg_idx];
		}
		else if (label == "FUNC2")
		{
			local_der = std::make_shared<MockFunctor>(
				teq::TensptrsT{op->get_args()[arg_idx]}, teq::Opcode{"FUNC4", 3});
		}
		else
		{
			local_der = teq::TensptrT(new MockLeaf(op->shape(), "other"));
		}
		teq::TensptrT tens(new MockFunctor(
			teq::TensptrsT{op,local_der}, teq::Opcode{"FUNC2", 1}));
		return teq::TensptrT(new MockFunctor(
			teq::TensptrsT{tens, supgrad}, teq::Opcode{"FUNC3", 2}));
	}

	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockLeaf(shape, "1"));
	}

	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		return teq::TensptrT(new MockLeaf(shape, "0"));
	}

	teq::TensptrT add (teq::TensptrsT elems) const override
	{
		return teq::TensptrT(new MockFunctor(elems, teq::Opcode{"FUNC", 0}));
	}
};


TEST(GRAD, OneZero)
{
	MockDeriveFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new MockLeaf(shape, "leaf"));
	teq::TensptrT leaf2(new MockLeaf(shape, "leaf2"));
	teq::TensptrT leaf3(new MockLeaf(shape, "leaf3"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf, leaf2}, teq::Opcode{"FUNC", 0}));

	auto wun = teq::derive(f, {f}, builder)[0];
	auto wun2 = teq::derive(leaf, {leaf}, builder)[0];
	auto wun3 = teq::derive(leaf3, {leaf3}, builder)[0];

	std::string shapestr = shape.to_string();
	EXPECT_STREQ("1", wun->to_string().c_str());
	EXPECT_STREQ("1", wun2->to_string().c_str());
	EXPECT_STREQ("1", wun3->to_string().c_str());

	auto zro = teq::derive(leaf, {leaf3}, builder)[0];
	auto zro2 = teq::derive(leaf3, {leaf}, builder)[0];
	auto zro3 = teq::derive(f, {leaf3}, builder)[0];
	auto zro4 = teq::derive(leaf, {nullptr}, builder)[0];
	auto zro5 = teq::derive(nullptr, {leaf}, builder)[0];

	EXPECT_STREQ("0", zro->to_string().c_str());
	EXPECT_STREQ("0", zro2->to_string().c_str());
	EXPECT_STREQ("0", zro3->to_string().c_str());
	EXPECT_STREQ("0", zro4->to_string().c_str());
	EXPECT_STREQ("0", zro5->to_string().c_str());
}


TEST(GRAD, BuilderStandardV)
{
	MockDeriveFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* standard v
	 *
	 * leaf  leaf2
	 *   \    /
	 *   FUNC
	 */
	teq::TensptrT leaf(new MockLeaf(shape, "leaf"));
	teq::TensptrT leaf2(new MockLeaf(shape, "leaf2"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf, leaf2}, teq::Opcode{"FUNC", 0}));

	auto gl = teq::derive(f, {leaf}, builder)[0];
	auto gl2 = teq::derive(f, {leaf2}, builder)[0];

	EXPECT_GRAPHEQ(
		"(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);

	EXPECT_GRAPHEQ(
		"(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(constant:leaf2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])",
		gl2);
}


TEST(GRAD, BuilderDiamond)
{
	MockDeriveFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond
	 *
	 *   leaf
	 *   /   \
	 * FUNC  FUNC2
	 *   \   /
	 *   FUNC3
	 */
	teq::TensptrT leaf(new MockLeaf(shape, "leaf"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f, f2}, teq::Opcode{"FUNC3", 2}));

	auto gl = teq::derive(f3, {leaf}, builder)[0];

	EXPECT_GRAPHEQ(
		"(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___`--(constant:other<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(constant:other<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);
}


TEST(GRAD, SymmetricalDiamond)
{
	MockDeriveFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond
	 *
	 *   leaf
	 *   /   \
	 * FUNC  FUNC
	 *   \   /
	 *   FUNC2
	 */
	teq::TensptrT leaf(new MockLeaf(shape, "leaf"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f, f2}, teq::Opcode{"FUNC2", 2}));

	auto gl = teq::derive(f3, {leaf}, builder)[0];

	EXPECT_GRAPHEQ(
		"(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|_______`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______|___________`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|_______`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|_______`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___________`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n",
		gl);
}


TEST(GRAD, TadPole)
{
	MockDeriveFunc builder;

	teq::DimsT slist = {94, 78, 70, 82, 62, 29, 38};
	teq::Shape shape(slist);

	/* diamond with a tail
	 *
	 *   leaf
	 *    |
	 *   FUNC
	 *   /   \
	 * FUNC2  FUNC3
	 *   \   /
	 *   FUNC4
	 */
	teq::TensptrT leaf(new MockLeaf(shape, "leaf"));
	teq::TensptrT f(new MockFunctor(teq::TensptrsT{leaf}, teq::Opcode{"FUNC", 0}));
	teq::TensptrT f2(new MockFunctor(teq::TensptrsT{f}, teq::Opcode{"FUNC2", 1}));
	teq::TensptrT f3(new MockFunctor(teq::TensptrsT{f}, teq::Opcode{"FUNC3", 2}));
	teq::TensptrT f4(new MockFunctor(teq::TensptrsT{f2, f3}, teq::Opcode{"FUNC4", 3}));

	auto gl = teq::derive(f4, {leaf}, builder)[0];
	EXPECT_GRAPHEQ(
		"(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_|___`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___|___`--(constant:other<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|_______`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___|___________`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______|___`--(constant:other<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____|_______`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|_______`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________|___________`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_________`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___`--(FUNC4<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|___`--(FUNC2<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|___|___`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|___|_______`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|___`--(FUNC3<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|_______`--(FUNC<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___|___________`--(constant:leaf<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________|___`--(constant:other<no_type>[94\\78\\70\\82\\62\\29\\38\\1])\n"
		"_____________`--(constant:1<no_type>[94\\78\\70\\82\\62\\29\\38\\1])",
		gl);
}


#endif // DISABLE_TEQ_GRAD_TEST
