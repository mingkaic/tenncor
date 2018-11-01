
#ifndef DISABLE_GRADER_TEST


#include <fstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "ade/functor.hpp"


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


TEST(GRADER, COPY)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::COPY, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/copy.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, ABS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::ABS, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/abs.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, NEG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::NEG, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/neg.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, SIN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::SIN, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/sin.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, COS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::COS, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/cos.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, TAN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::TAN, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/tan.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, EXP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::EXP, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/exp.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, LOG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::LOG, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/log.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, SQRT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::SQRT, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/sqrt.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, ROUND)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::ROUND, {{ade::identity, leaf}});
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd->gradient(leaf.get());

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);

	std::ifstream ostr(testdir + "/round.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g1);
}


TEST(GRADER, POW)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::POW, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::POW, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/pow1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/powl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/powr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, ADD)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::ADD, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::ADD, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/add1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/addl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/addr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, SUB)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::SUB, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::SUB, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/sub1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/subl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/subr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, MUL)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::MUL, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::MUL, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/mul1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/mull.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/mulr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, DIV)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::DIV, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::DIV, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/div1.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/divl.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/divr.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, EQ)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::EQ, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::EQ, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, NE)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::NE, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::NE, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, LT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::LT, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::LT, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, GT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::GT, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::GT, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, RAND_BINO)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::RAND_BINO, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::RAND_BINO, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, RAND_UNIF)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::RAND_UNIF, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::RAND_UNIF, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


TEST(GRADER, RAND_NORM)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(ade::RAND_NORM, {
		{ade::identity, leaf}, {ade::identity, leaf}});
	ade::Tensorptr fwd2 = ade::Functor::get(ade::RAND_NORM, {
		{ade::identity, leaf}, {ade::identity, leaf1}});
	ade::Tensorptr g = fwd2->gradient(leaf1.get());
	ade::Tensorptr g1 = fwd2->gradient(leaf.get());
	ade::Tensorptr g2 = fwd->gradient(leaf.get());
	ade::Tensorptr g0 = fwd->gradient(leaf1.get());

	std::ifstream ostr(testdir + "/grad0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(ostr, g2);

	std::ifstream lstr(testdir + "/0.txt");
	ASSERT_TRUE(ostr.is_open());
	TREE_EQ(lstr, g);

	std::ifstream rstr(testdir + "/0.txt");
	ASSERT_TRUE(rstr.is_open());
	TREE_EQ(rstr, g1);

	std::ifstream zstr(testdir + "/0.txt");
	ASSERT_TRUE(zstr.is_open());
	TREE_EQ(zstr, g0);
}


#endif // DISABLE_GRADER_TEST
