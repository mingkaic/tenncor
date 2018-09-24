#include <fstream>

#include "gtest/gtest.h"

#include "cli/util/ascii_tree.hpp"

#include "ade/functor.hpp"
#include "ade/grader.hpp"


#ifndef DISABLE_GRADER_TEST


const std::string testdir = "ade/test/data";


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


static void TREE_EQ (std::istream& expectstr, ade::iTensor* root)
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

#if 0
	std::cout << gotstr.str() << std::endl;
#endif

	std::string expect;
	std::string got;
	std::string line;
	while (std::getline(expectstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			expect += line + "\n";
		}
	}
	while (std::getline(gotstr, line))
	{
		trim(line);
		if (line.size() > 0)
		{
			got += line + "\n";
		}
	}
	EXPECT_STREQ(expect.c_str(), got.c_str());
}


TEST(GRADER, ABS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ABS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ABS>({leaf}, leaf);

	std::ifstream zstr(testdir + "/abs0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/abs1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, NEG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NEG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NEG>({leaf}, leaf);

	std::ifstream zstr(testdir + "/neg0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/neg1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, NOT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NOT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NOT>({leaf}, leaf);

	std::ifstream zstr(testdir + "/not0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/not1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, SIN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SIN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SIN>({leaf}, leaf);

	std::ifstream zstr(testdir + "/sin0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/sin1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, COS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::COS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::COS>({leaf}, leaf);

	std::ifstream zstr(testdir + "/cos0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/cos1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, TAN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::TAN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::TAN>({leaf}, leaf);

	std::ifstream zstr(testdir + "/tan0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/tan1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, EXP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::EXP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::EXP>({leaf}, leaf);

	std::ifstream zstr(testdir + "/exp0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/exp1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, LOG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::LOG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::LOG>({leaf}, leaf);

	std::ifstream zstr(testdir + "/log0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/log1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, SQRT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SQRT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SQRT>({leaf}, leaf);

	std::ifstream zstr(testdir + "/sqrt0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/sqrt1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, ROUND)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ROUND>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ROUND>({leaf}, leaf);

	std::ifstream zstr(testdir + "/round0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/round1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, FLIP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::FLIP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::FLIP>({leaf}, leaf);

	std::ifstream zstr(testdir + "/zero.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/one.txt");
	ASSERT_TRUE(ostr.is_open());
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

	std::ifstream ostr(testdir + "/pow1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/powl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/powr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/add1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/addl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/addr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/sub1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/subl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/subr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/mul1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/mull.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/mulr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/div1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/divl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/divr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/eq1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/eql.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/eqr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/ne1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/nel.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/ner.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/lt1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/ltl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/ltr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/gt1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/gtl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/gtr.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
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

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2.get());

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g.get());

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1.get());
}


TEST(GRADER, N_ELEMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_ELEMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_ELEMS>({leaf}, leaf);

	std::ifstream zstr(testdir + "/zero.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, N_DIMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_DIMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_DIMS>({leaf}, leaf);

	std::ifstream zstr(testdir + "/zero.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
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

	std::ifstream zstr(testdir + "/rmax0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/rmax1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


TEST(GRADER, RSUM)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RSUM>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RSUM>({leaf}, leaf);

	std::ifstream zstr(testdir + "/rsum0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/rsum1.txt");
	ASSERT_TRUE(ostr.is_open());
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

	std::ifstream lstr(testdir + "/matmula.txt");
	ASSERT_TRUE(lstr.is_open());
	TREE_EQ(lstr, ga.get());

	std::ifstream rstr(testdir + "/matmulb.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, gb.get());

	std::ifstream zlstr(testdir + "/matmula0.txt");
	ASSERT_TRUE(zlstr.is_open());
	TREE_EQ(zlstr, z.get());

	std::ifstream zrstr(testdir + "/matmulb0.txt");
	ASSERT_TRUE(zrstr.is_open());
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

	std::ifstream big_lstr(testdir + "/big_matmula.txt");
	ASSERT_TRUE(big_lstr.is_open());
	TREE_EQ(big_lstr, g1a1.get());

	std::ifstream big_rstr(testdir + "/big_matmulb.txt");
	ASSERT_TRUE(big_rstr.is_open());
	TREE_EQ(big_rstr, g1b1.get());

	std::ifstream g2a_str(testdir + "/matmul_g2a.txt");
	ASSERT_TRUE(g2a_str.is_open());
	TREE_EQ(g2a_str, g2a.get());

	std::ifstream g2a1_str(testdir + "/matmul_g2a1.txt");
	ASSERT_TRUE(g2a1_str.is_open());
	TREE_EQ(g2a1_str, g2a1.get());

	std::ifstream g3b_str(testdir + "/matmul_g3b.txt");
	ASSERT_TRUE(g3b_str.is_open());
	TREE_EQ(g3b_str, g3b.get());

	std::ifstream g3b1_str(testdir + "/matmul_g3b1.txt");
	ASSERT_TRUE(g3b1_str.is_open());
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

	std::ifstream zstr(testdir + "/perm0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/perm1.txt");
	ASSERT_TRUE(ostr.is_open());
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

	std::ifstream zstr(testdir + "/extend0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/extend1.txt");
	ASSERT_TRUE(ostr.is_open());
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

	std::ifstream zstr(testdir + "/reshape0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0.get());

	std::ifstream ostr(testdir + "/reshape1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1.get());
}


#endif /* DISABLE_GRADER_TEST */
