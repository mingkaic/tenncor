
#ifndef DISABLE_FILTER_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "opt/filter.hpp"


// TEST(FILTER, RemoveDuplicates)
// {
// 	teq::Shape shape({2, 3, 4});
// 	teq::TensptrT var(new MockTensor(shape, "special_var", false));
// 	teq::TensptrT zero(new MockTensor(shape, "0"));
// 	teq::TensptrT one(new MockTensor(shape, "1"));
// 	teq::TensptrT two(new MockTensor(shape, "2"));

// 	teq::TensptrsT roots;
// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = zero + got1 * got22;
// 		auto got11 = tenncor::pow(got2, zero);

// 		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
// 		roots.push_back(tenncor::pow(m, got3 / gotn1) - got2);
// 	}

// 	{
// 		auto other_got1 = tenncor::cos(zero);
// 		auto got22 = tenncor::max(two, zero);
// 		roots.push_back(other_got1 * got22);
// 	}

// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = zero + got1 * got22;
// 		auto got11 = tenncor::pow(got2, zero);

// 		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
// 		roots.push_back(tenncor::pow(m, got3 / gotn1) - got2);
// 	}

// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = got2 / (got1 * got22);
// 		auto got11 = too == gotn1;

// 		roots.push_back((got11 * got1) * (too * got3));
// 	}

// 	opt::remove_duplicates(roots, mock_equals);
// 	ASSERT_EQ(4, roots.size());
// 	auto opt_subroot = roots[0];
// 	auto opt_root = roots[1];
// 	auto opt_splitroot = roots[2];
// 	auto opt_copyroot = roots[3];

// 	ASSERT_NE(nullptr, opt_subroot);
// 	ASSERT_NE(nullptr, opt_root);
// 	ASSERT_NE(nullptr, opt_splitroot);
// 	ASSERT_NE(nullptr, opt_copyroot);

// 	std::stringstream ss;
// 	CSVEquation ceq;
// 	opt_subroot->accept(ceq);
// 	opt_root->accept(ceq);
// 	opt_splitroot->accept(ceq);
// 	opt_copyroot->accept(ceq);
// 	ceq.to_stream(ss);

// 	std::list<std::string> expectlines =
// 	{
// 		"0:MUL,1:COS,0,white",
// 		"0:MUL,3:MAX,1,white",
// 		"10:ADD,0:MUL,1,white",
// 		"10:ADD,2:0,0,white",
// 		"11:POW,12:SUB,0,white",
// 		"11:POW,2:0,1,white",
// 		"12:SUB,2:0,1,white",
// 		"12:SUB,4:2,0,white",
// 		"13:DIV,14:ADD,0,white",
// 		"13:DIV,17:SUB,1,white",
// 		"14:ADD,15:ADD,0,white",
// 		"14:ADD,4:2,1,white",
// 		"15:ADD,16:1,0,white",
// 		"15:ADD,2:0,1,white",
// 		"17:SUB,16:1,1,white",
// 		"17:SUB,2:0,0,white",
// 		"18:MUL,19:MUL,0,white",
// 		"18:MUL,22:MUL,1,white",
// 		"19:MUL,1:COS,1,white",
// 		"19:MUL,20:EQ,0,white",
// 		"1:COS,2:0,0,white",
// 		"20:EQ,17:SUB,1,white",
// 		"20:EQ,21:DIV,0,white",
// 		"21:DIV,0:MUL,1,white",
// 		"21:DIV,12:SUB,0,white",
// 		"22:MUL,14:ADD,1,white",
// 		"22:MUL,21:DIV,0,white",
// 		"3:MAX,2:0,1,white",
// 		"3:MAX,4:2,0,white",
// 		"5:SUB,12:SUB,1,white",
// 		"5:SUB,6:POW,0,white",
// 		"6:POW,13:DIV,1,white",
// 		"6:POW,7:MIN,0,white",
// 		"7:MIN,8:MIN,0,white",
// 		"7:MIN,9:MIN,1,white",
// 		"8:MIN,1:COS,1,white",
// 		"8:MIN,3:MAX,0,white",
// 		"9:MIN,10:ADD,0,white",
// 		"9:MIN,11:POW,1,white",
// 	};
// 	expectlines.sort();
// 	std::list<std::string> gotlines;
// 	std::string line;
// 	while (std::getline(ss, line))
// 	{
// 		gotlines.push_back(line);
// 	}
// 	gotlines.sort();
// 	std::vector<std::string> diffs;
// 	std::set_symmetric_difference(expectlines.begin(), expectlines.end(),
// 		gotlines.begin(), gotlines.end(), std::back_inserter(diffs));

// 	std::stringstream diffstr;
// 	for (auto diff : diffs)
// 	{
// 		diffstr << diff << "\n";
// 	}
// 	std::string diffmsg = diffstr.str();
// 	EXPECT_EQ(0, diffmsg.size()) << "mismatching edges:\n" << diffmsg;
// }


static teq::TensptrT mock_constants (teq::FuncptrT f)
{
	std::string opname = f->get_opcode().name_;
	if (opname == "COS")
	{
		return std::make_shared<MockTensor>(f->shape(), "5");
	}
	else if (opname == "POW")
	{
		return std::make_shared<MockTensor>(f->shape(), "6");
	}
	else if (opname == "SUB")
	{
		return std::make_shared<MockTensor>(f->shape(), "7");
	}
	[]()
	{
		FAIL() << "should not convert constant for functor with non-constant arguments";
	}();
	return f;
}


TEST(FILTER, ConstantFuncs)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockTensor(shape, "special_var", false));
	teq::TensptrT two(new MockTensor(shape, "2"));
	teq::TensptrT three(new MockTensor(shape, "3"));
	teq::TensptrT four(new MockTensor(shape, "4"));

	teq::TensptrsT roots;
	roots.reserve(3);

	auto vfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"SIN", 0});
	roots.push_back(vfunc);
	auto cfunc = std::make_shared<MockFunctor>(teq::TensptrsT{two}, teq::Opcode{"COS", 0});
	roots.push_back(cfunc);
	auto adv_func = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{
			std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"SIN", 0}),
			std::make_shared<MockFunctor>(teq::TensptrsT{three, four}, teq::Opcode{"POW", 0}),
		}, teq::Opcode{"ADD", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{
			std::make_shared<MockFunctor>(teq::TensptrsT{two}, teq::Opcode{"COS", 0}),
			two
		}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"MUL", 0});
	roots.push_back(adv_func);
	opt::constant_funcs(roots, mock_constants);
	ASSERT_EQ(3, roots.size());
	auto opt_vfunc = roots[0];
	auto opt_cfunc = roots[1];
	auto opt_adv_func = roots[2];
	ASSERT_NE(nullptr, opt_vfunc);
	ASSERT_NE(nullptr, opt_cfunc);
	ASSERT_NE(nullptr, opt_adv_func);

	// expect optimized vfunc to remain the same
	EXPECT_GRAPHEQ(
		"(SIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(variable:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
		opt_vfunc);
	// expect optimized cfunc to be constant 5
	EXPECT_GRAPHEQ(
		"(constant:5[2\\3\\4\\1\\1\\1\\1\\1])",
		opt_cfunc);
	// expect optimized adv_func to be (sin(var) + constant 6) * (constant 7)
	EXPECT_GRAPHEQ(
		"(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(ADD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(SIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(variable:special_var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(constant:6[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:7[2\\3\\4\\1\\1\\1\\1\\1])",
		opt_adv_func);
}


#endif // DISABLE_FILTER_TEST
