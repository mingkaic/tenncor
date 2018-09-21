#include <fstream>

#include "gtest/gtest.h"

#include "cli/util/ascii_tree.hpp"

#include "ade/functor.hpp"
#include "ade/grader.hpp"


#ifndef DISABLE_GRADER_TEST


static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int, int>(std::isspace))));
}


static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}


static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}


void TREE_EQ (std::istream& expectstr, ade::iTensor* root)
{
	PrettyTree<ade::iTensor*> artist(
		[](ade::iTensor*& root) -> std::vector<ade::iTensor*>
		{
			if (ade::iFunctor* f = dynamic_cast<ade::iFunctor*>(root))
			{
				return f->get_refs();
			}
			return {};
		},
		[](std::ostream& out, ade::iTensor*& root)
		{
			if (root)
			{
				if (root == ade::Tensor::SYMBOLIC_ONE.get())
				{
					out << 1;
				}
				else if (root == ade::Tensor::SYMBOLIC_ZERO.get())
				{
					out << 0;
				}
				else
				{
					out << root->to_string();
				}
			}
		});

	std::stringstream gotstr;
	artist.print(gotstr, root);

#ifdef _DEBUG_MODE
	std::cout << gotstr.str() << std::endl;
#endif /* _DEBUG_MODE */

	std::string expect;
	std::string got;
	std::string expectline;
	std::string gotline;
	size_t linum = 0;
	while (std::getline(expectstr, expectline) &&
		std::getline(gotstr, gotline))
	{
		trim(expectline);
		trim(gotline);
		EXPECT_STREQ(expectline.c_str(), gotline.c_str()) <<
			"line number: " << linum;
		expect += expectline + "\n";
		got += gotline + "\n";
		++linum;
	}
	while (std::getline(expectstr, expectline))
	{
		trim(expectline);
		if (expectline.size() > 0)
		{
			expect += expectline + "\n";
			FAIL() << "tree compare prematurely ended at line " << linum << "\n" <<
				"expected:\n======\n" << expect << "======\n" <<
				"got:\n======\n" << got << "======\n";
		}
	}
	while (std::getline(gotstr, gotline))
	{
		trim(gotline);
		if (gotline.size() > 0)
		{
			got += gotline + "\n";
			FAIL() << "tree compare prematurely ended at line " << linum << "\n" <<
				"expected:\n======\n" << expect << "======\n" <<
				"got:\n======\n" << got << "======\n";
		}
	}
}


TEST(GRADER, ABS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ABS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ABS>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/abs0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/abs1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, NEG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NEG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NEG>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/neg0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/neg1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, NOT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NOT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NOT>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/not0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/not1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, SIN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SIN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SIN>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/sin0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/sin1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, COS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::COS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::COS>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/cos0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/cos1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, TAN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::TAN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::TAN>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/tan0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/tan1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, EXP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::EXP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::EXP>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/exp0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/exp1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, LOG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::LOG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::LOG>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/log0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/log1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, SQRT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SQRT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SQRT>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/sqrt0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/sqrt1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, ROUND)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ROUND>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ROUND>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/round0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/round1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, FLIP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::FLIP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::FLIP>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/one.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, POW)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::POW>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::POW>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::POW>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/pow1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/powl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/powr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, ADD)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::ADD>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ADD>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::ADD>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/add1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/addl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/addr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, SUB)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::SUB>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SUB>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::SUB>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/sub1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/subl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/subr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, MUL)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::MUL>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::MUL>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::MUL>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/mul1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/mull.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/mulr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, DIV)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::DIV>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::DIV>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::DIV>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/div1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/divl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/divr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, EQ)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::EQ>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::EQ>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::EQ>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/eq1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/eql.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/eqr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, NE)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::NE>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NE>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::NE>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/ne1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/nel.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/ner.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, LT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::LT>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::LT>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::LT>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/lt1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/ltl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/ltr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, GT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::GT>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::GT>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::GT>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/gt1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/gtl.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/gtr.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, BINO)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::BINO>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::BINO>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::BINO>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, UNIF)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::UNIF>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::UNIF>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::UNIF>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, NORM)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::NORM>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NORM>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::NORM>({leaf, leaf}, leaf);

	std::ifstream ostr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, N_ELEMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_ELEMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_ELEMS>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, N_DIMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_DIMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_DIMS>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/zero.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/zero.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, ARGMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	EXPECT_THROW(ade::grader<ade::ARGMAX>({leaf}, leaf1), std::bad_function_call);
	EXPECT_THROW(ade::grader<ade::ARGMAX>({leaf}, leaf), std::bad_function_call);
}


TEST(GRADER, RMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RMAX>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RMAX>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/rmax0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/rmax1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, RSUM)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RSUM>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RSUM>({leaf}, leaf);

	std::ifstream zstr("ade/test/expects/rsum0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/rsum1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, MATMUL)
{
	std::vector<ade::DimT> alist = {2, 3};
	std::vector<ade::DimT> blist = {4, 2};
	ade::Tensorptr a = ade::Tensor::get(ade::Shape(alist));
	ade::Tensorptr b = ade::Tensor::get(ade::Shape(blist));

	std::vector<ade::DimT> alist1 = {4, 2, 3};
	std::vector<ade::DimT> blist1 = {3, 4, 2};
	ade::Tensorptr a1 = ade::Tensor::get(ade::Shape(alist1));
	ade::Tensorptr b1 = ade::Tensor::get(ade::Shape(blist1));

	ade::Tensorptr ga = ade::grader<ade::MATMUL>({a, b}, a);
	ade::Tensorptr gb = ade::grader<ade::MATMUL>({a, b}, b);
	ade::Tensorptr z = ade::grader<ade::MATMUL>({a, b}, a1);
	ade::Tensorptr z1 = ade::grader<ade::MATMUL>({a, b}, b1);

	std::ifstream lstr("ade/test/expects/matmula.txt");
	EXPECT_TRUE(lstr.is_open());
	TREE_EQ(lstr, ga.get());

	std::ifstream rstr("ade/test/expects/matmulb.txt");
	EXPECT_TRUE(rstr.is_open());
	TREE_EQ(rstr, gb.get());

	std::ifstream zlstr("ade/test/expects/matmula0.txt");
	EXPECT_TRUE(zlstr.is_open());
	TREE_EQ(zlstr, z.get());

	std::ifstream zrstr("ade/test/expects/matmulb0.txt");
	EXPECT_TRUE(zrstr.is_open());
	TREE_EQ(zrstr, z1.get());

	ade::Tensorptr g1a1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, a1, 2, 1);
	ade::Tensorptr g1b1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, b1, 2, 1);
	ade::Tensorptr g2a = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, a, 2, 1);
	ade::Tensorptr g2a1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, a1, 2, 1);
	ade::Tensorptr g3b = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, b, 2, 1);
	ade::Tensorptr g3b1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, b1, 2, 1);

	std::ifstream big_lstr("ade/test/expects/big_matmula.txt");
	EXPECT_TRUE(big_lstr.is_open());
	TREE_EQ(big_lstr, g1a1.get());

	std::ifstream big_rstr("ade/test/expects/big_matmulb.txt");
	EXPECT_TRUE(big_rstr.is_open());
	TREE_EQ(big_rstr, g1b1.get());

	std::ifstream g2a_str("ade/test/expects/matmul_g2a.txt");
	EXPECT_TRUE(g2a_str.is_open());
	TREE_EQ(g2a_str, g2a.get());

	std::ifstream g2a1_str("ade/test/expects/matmul_g2a1.txt");
	EXPECT_TRUE(g2a1_str.is_open());
	TREE_EQ(g2a1_str, g2a1.get());

	std::ifstream g3b_str("ade/test/expects/matmul_g3b.txt");
	EXPECT_TRUE(g3b_str.is_open());
	TREE_EQ(g3b_str, g3b.get());

	std::ifstream g3b1_str("ade/test/expects/matmul_g3b1.txt");
	EXPECT_TRUE(g3b1_str.is_open());
	TREE_EQ(g3b1_str, g3b1.get());

	std::vector<ade::DimT> duplist = {3, 3};
	ade::Tensorptr d = ade::Tensor::get(ade::Shape(duplist));
	ade::Tensorptr gd = ade::grader<ade::MATMUL>({d, d}, d);
}


TEST(GRADER, PERMUTE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, leaf1, {2, 1, 3, 4, 0});
	ade::Tensorptr g1 = ade::grader<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, leaf, {2, 1, 3, 4, 0});

	std::ifstream zstr("ade/test/expects/perm0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/perm1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, EXTEND)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, leaf1, {4});
	ade::Tensorptr g1 = ade::grader<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, leaf, {4});

	std::ifstream zstr("ade/test/expects/extend0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/extend1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, RESHAPE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	std::vector<ade::DimT> olist = {2, 12, 5, 7};

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RESHAPE,
		std::vector<ade::DimT>>({leaf}, leaf1, olist);
	ade::Tensorptr g1 = ade::grader<ade::RESHAPE,
		std::vector<ade::DimT>>({leaf}, leaf, olist);

	std::ifstream zstr("ade/test/expects/reshape0.txt");
	EXPECT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr("ade/test/expects/reshape1.txt");
	EXPECT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


#endif /* DISABLE_GRADER_TEST */
