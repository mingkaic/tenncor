#include <fstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "ade/functor.hpp"
#include "ade/grader.hpp"


#ifndef DISABLE_GRADER_TEST


const std::string testdir = "ade/test/data";


static inline void ltrim(std::string &s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(),
		std::not1(std::ptr_fun<int,int>(std::isspace))));
}


static inline void rtrim(std::string &s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(),
		std::not1(std::ptr_fun<int,int>(std::isspace))).base(), s.end());
}


static inline void trim(std::string &s)
{
	ltrim(s);
	rtrim(s);
}


static void TREE_EQ (std::istream& expectstr, ade::Tensorptr& root)
{
	PrettyEquation artist;
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/abs1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/neg1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/not1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/sin1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/cos1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/tan1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/exp1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/log1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/sqrt1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/round1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, FLIP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::FLIP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::FLIP>({leaf}, leaf);

	std::ifstream zstr(testdir + "/flip0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/one.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/powl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/powr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/addl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/addr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/subl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/subr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/mull.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/mulr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/divl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/divr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/eql.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/eqr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/nel.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/ner.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/ltl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/ltr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/gtl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/gtr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
}


TEST(GRADER, RAND_BINO)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::RAND_BINO>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RAND_BINO>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::RAND_BINO>({leaf, leaf}, leaf);

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
}


TEST(GRADER, RAND_UNIF)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::RAND_UNIF>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RAND_UNIF>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::RAND_UNIF>({leaf, leaf}, leaf);

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
}


TEST(GRADER, RAND_NORM)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::RAND_NORM>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::RAND_NORM>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::RAND_NORM>({leaf, leaf}, leaf);

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/zero.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/zero.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, ARGMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	auto fail = [&](){ ade::grader<ade::ARGMAX,uint8_t>({leaf}, leaf1, 8); };
	auto fail2 = [&]() { ade::grader<ade::ARGMAX,uint8_t>({leaf}, leaf, 8); };
	EXPECT_THROW(fail(), std::bad_function_call);
	EXPECT_THROW(fail2(), std::bad_function_call);
}


TEST(GRADER, RMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RMAX,uint8_t>({leaf}, leaf1, 8);
	ade::Tensorptr g1 = ade::grader<ade::RMAX,uint8_t>({leaf}, leaf, 8);

	std::ifstream zstr(testdir + "/rmax0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/rmax1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, RSUM)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RSUM,uint8_t>({leaf}, leaf1, 8);
	ade::Tensorptr g1 = ade::grader<ade::RSUM,uint8_t>({leaf}, leaf, 8);

	std::ifstream zstr(testdir + "/rsum0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/rsum1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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

	ade::Tensorptr ga = ade::grader<ade::MATMUL,uint8_t,uint8_t>(
		{a, b}, a, 1, 1);
	ade::Tensorptr gb = ade::grader<ade::MATMUL,uint8_t,uint8_t>(
		{a, b}, b, 1, 1);
	ade::Tensorptr z = ade::grader<ade::MATMUL,uint8_t,uint8_t>(
		{a, b}, a1, 1, 1);
	ade::Tensorptr z1 = ade::grader<ade::MATMUL,uint8_t,uint8_t>(
		{a, b}, b1, 1, 1);

	std::ifstream lstr(testdir + "/matmula.txt");
	ASSERT_TRUE(lstr.is_open());
	TREE_EQ(lstr, ga);

	std::ifstream rstr(testdir + "/matmulb.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, gb);

	std::ifstream zlstr(testdir + "/matmula0.txt");
	ASSERT_TRUE(zlstr.is_open());
	TREE_EQ(zlstr, z);

	std::ifstream zrstr(testdir + "/matmulb0.txt");
	ASSERT_TRUE(zrstr.is_open());
	TREE_EQ(zrstr, z1);

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
	TREE_EQ(big_lstr, g1a1);

	std::ifstream big_rstr(testdir + "/big_matmulb.txt");
	ASSERT_TRUE(big_rstr.is_open());
	TREE_EQ(big_rstr, g1b1);

	std::ifstream g2a_str(testdir + "/matmul_g2a.txt");
	ASSERT_TRUE(g2a_str.is_open());
	TREE_EQ(g2a_str, g2a);

	std::ifstream g2a1_str(testdir + "/matmul_g2a1.txt");
	ASSERT_TRUE(g2a1_str.is_open());
	TREE_EQ(g2a1_str, g2a1);

	std::ifstream g3b_str(testdir + "/matmul_g3b.txt");
	ASSERT_TRUE(g3b_str.is_open());
	TREE_EQ(g3b_str, g3b);

	std::ifstream g3b1_str(testdir + "/matmul_g3b1.txt");
	ASSERT_TRUE(g3b1_str.is_open());
	TREE_EQ(g3b1_str, g3b1);

	std::vector<ade::DimT> duplist = {3, 3};
	ade::Tensorptr d = ade::Tensor::get(ade::Shape(duplist));
	ade::Tensorptr gd = ade::grader<ade::MATMUL,uint8_t,uint8_t>(
		{d, d}, d, 1, 1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/perm1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/extend1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
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
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/reshape1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


#endif // DISABLE_GRADER_TEST
