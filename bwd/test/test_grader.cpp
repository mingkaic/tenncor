
#ifndef DISABLE_GRADER_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "dbg/ade.hpp"

#include "bwd/grader.hpp"


struct MockTensor final : public ade::Tensor
{
	MockTensor (void) = default;

	MockTensor (ade::Shape shape) :
		str_(shape.n_elems(), 0), shape_(shape) {}

	/// Implementation of iTensor
	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	/// Implementation of iTensor
	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	char* data (void) override
	{
		return &str_[0];
	}

	const char* data (void) const override
	{
		return str_.c_str();
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	std::string str_;

	ade::Shape shape_;
};


struct MockRuleSet final : public age::iRuleSet
{
	ade::Tensor* data (double scalar, ade::Shape shape) override
	{
		auto out = new ::MockTensor(shape);
		std::fill(out->str_.begin(), out->str_.end(), (int) scalar);
		return out;
	}

	ade::Opcode sum_opcode (void) override
	{
		return ade::Opcode{"+", 0};
	}

	ade::Opcode prod_opcode (void) override
	{
		return ade::Opcode{"*", 1};
	}

	ade::Tensorptr grad_rule (size_t code, age::TensT args, size_t idx) override
	{
		// grad of sum is prod and grad of prod is sum
		if (code)
		{
			return ade::Functor::get(sum_opcode(), age::to_args(args));
		}
		return ade::Functor::get(prod_opcode(), age::to_args(args));
	}
};


std::unique_ptr<age::iRuleSet> age::Grader::rules_ =
	std::make_unique<MockRuleSet>();


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


TEST(GRADER, SUM)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr outside = new MockTensor(ade::Shape({7}));
	ade::Tensorptr leaf = new MockTensor(ade::Shape(slist));
	ade::Tensorptr leaf1 = new MockTensor(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(
		age::Grader::rules_->sum_opcode(), {
		{ade::identity, leaf},
		{ade::identity, leaf1},
	});

	ade::Tensorptr g0 = age::derive(fwd, outside.get());
	ade::Tensorptr gl = age::derive(fwd, leaf.get());
	ade::Tensorptr gr = age::derive(fwd, leaf1.get());

	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	zstr << "([7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(+)\n" <<
		" `--(*)\n" <<
		"     `--(*)\n" << // chain rule (derivative of SUM is PROD)
		"     |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     |   `--(+)\n" <<
		"     |       `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+)\n" << // derivative of leaf wrt leaf
		"         `--(+)\n" <<
		"             `--([2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(+)\n" <<
		" `--(*)\n" <<
		"     `--(*)\n" << // chain rule
		"     |   `--(+)\n" <<
		"     |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+)\n" << // derivative of leaf wrt leaf
		"         `--(+)\n" <<
		"             `--([2\\3\\1\\1\\1\\1\\1\\1])\n";

	TREE_EQ(zstr, g0);
	TREE_EQ(lstr, gl);
	TREE_EQ(rstr, gr);
}


TEST(GRADER, PROD)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr outside = new MockTensor(ade::Shape({7}));
	ade::Tensorptr leaf = new MockTensor(ade::Shape(slist));
	ade::Tensorptr leaf1 = new MockTensor(ade::Shape(slist));

	ade::Tensorptr fwd = ade::Functor::get(
		age::Grader::rules_->prod_opcode(), {
		{ade::identity, leaf},
		{ade::identity, leaf1},
	});

	ade::Tensorptr g0 = age::derive(fwd, outside.get());
	ade::Tensorptr gl = age::derive(fwd, leaf.get());
	ade::Tensorptr gr = age::derive(fwd, leaf1.get());

	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	zstr << "([7\\1\\1\\1\\1\\1\\1\\1])\n";
	lstr <<
		"(+)\n" <<
		" `--(*)\n" <<
		"     `--(+)\n" << // chain rule (derivative of PROD is SUM)
		"     |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     |   `--(+)\n" <<
		"     |       `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+)\n" << // derivative of leaf wrt leaf
		"         `--(+)\n" <<
		"             `--([2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(+)\n" <<
		" `--(*)\n" <<
		"     `--(+)\n" << // chain rule
		"     |   `--(+)\n" <<
		"     |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"     `--(+)\n" << // derivative of leaf wrt leaf
		"         `--(+)\n" <<
		"             `--([2\\3\\1\\1\\1\\1\\1\\1])\n";

	TREE_EQ(zstr, g0);
	TREE_EQ(lstr, gl);
	TREE_EQ(rstr, gr);
}


#endif // DISABLE_GRADER_TEST
