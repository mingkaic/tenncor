
#ifndef DISABLE_GRADER_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "dbg/ade.hpp"

#include "bwd/grader.hpp"


struct MockTensor final : public ade::iLeaf
{
	MockTensor (void) = default;

	MockTensor (ade::Shape shape) : shape_(shape) {}

	const ade::Shape& shape (void) const override
	{
		return shape_;
	}

	std::string to_string (void) const override
	{
		return shape_.to_string();
	}

	void* data (void) override
	{
		return &val_;
	}

	const void* data (void) const override
	{
		return &val_;
	}

	size_t type_code (void) const override
	{
		return 0;
	}

	double val_;

	ade::Shape shape_;
};


struct MockRuleSet final : public age::iRuleSet
{
	ade::iLeaf* data (double scalar, ade::Shape shape) override
	{
		auto out = new ::MockTensor(shape);
		out->val_ = scalar;
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

	ade::TensptrT grad_rule (size_t code, age::TensT args, size_t idx) override
	{
		// grad of sum is prod and grad of prod is sum
		if (code)
		{
			return ade::TensptrT(ade::Functor::get(sum_opcode(), age::to_args(args)));
		}
		return ade::TensptrT(ade::Functor::get(prod_opcode(), age::to_args(args)));
	}
};


static std::shared_ptr<age::iRuleSet> mock_rules =
	std::make_shared<MockRuleSet>();


ade::TensptrT derive (ade::TensptrT& root, const ade::iTensor* wrt)
{
	age::Grader grader(wrt, mock_rules);
	root->accept(grader);
	auto it = grader.derivatives_.find(root.get());
	assert(grader.derivatives_.end() != it);
	return it->second;
}


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


static void TREE_EQ (std::istream& expectstr, ade::TensptrT& root)
{
	PrettyEquation artist;
	std::stringstream gotstr;
	artist.print(gotstr, root);

#if 0
	std::cout << gotstr.str() << '\n';
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


TEST(GRADER, Ruleset)
{
	ade::TensptrT tens(new MockTensor());

	EXPECT_FATAL(age::Grader(nullptr, mock_rules), "cannot derive with respect to null");
	EXPECT_FATAL(age::Grader(tens.get(), nullptr), "cannot derive without ruleset");
}


TEST(GRADER, Leaf)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT g1(derive(leaf, leaf.get()));
	ade::TensptrT g0(derive(leaf, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	EXPECT_NE(nullptr, mock1);
	EXPECT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream sstr;
	sstr << "([2\\3\\1\\1\\1\\1\\1\\1])\n";
	TREE_EQ(sstr, g1);
	sstr.clear();
	sstr << "([2\\3\\1\\1\\1\\1\\1\\1])\n";
	TREE_EQ(sstr, g0);
}


TEST(GRADER, Sum)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT fwd(ade::Functor::get(
		mock_rules->sum_opcode(), {
		{ade::identity, leaf},
		{ade::identity, leaf1},
	}));

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	EXPECT_NE(nullptr, mock1);
	EXPECT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "([2\\3\\1\\1\\1\\1\\1\\1])\n";
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

	TREE_EQ(ostr, g1);
	TREE_EQ(zstr, g0);
	TREE_EQ(lstr, gl);
	TREE_EQ(rstr, gr);
}


TEST(GRADER, Prod)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT fwd(ade::Functor::get(
		mock_rules->prod_opcode(), {
		{ade::identity, leaf},
		{ade::identity, leaf1},
	}));

	ade::TensptrT g1(derive(fwd, fwd.get()));
	ade::TensptrT g0(derive(fwd, outside.get()));
	ade::TensptrT gl(derive(fwd, leaf.get()));
	ade::TensptrT gr(derive(fwd, leaf1.get()));

	auto mock1 = dynamic_cast<MockTensor*>(g1.get());
	auto mock0 = dynamic_cast<MockTensor*>(g0.get());

	EXPECT_NE(nullptr, mock1);
	EXPECT_NE(nullptr, mock0);

	EXPECT_EQ(1, mock1->val_);
	EXPECT_EQ(0, mock0->val_);

	std::stringstream ostr;
	std::stringstream zstr;
	std::stringstream lstr;
	std::stringstream rstr;

	ostr << "([2\\3\\1\\1\\1\\1\\1\\1])\n";
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

	TREE_EQ(ostr, g1);
	TREE_EQ(zstr, g0);
	TREE_EQ(lstr, gl);
	TREE_EQ(rstr, gr);
}


TEST(GRADER, SumProd)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::TensptrT outside(new MockTensor(ade::Shape({7})));
	ade::TensptrT leaf(new MockTensor(ade::Shape(slist)));
	ade::TensptrT leaf1(new MockTensor(ade::Shape(slist)));

	ade::TensptrT prod(ade::Functor::get(
		mock_rules->prod_opcode(), {
		{ade::identity, leaf},
		{ade::identity, leaf1},
	}));

	ade::TensptrT sum(ade::Functor::get(
		mock_rules->sum_opcode(), {
		{ade::identity, prod},
		{ade::identity, prod},
	}));

	ade::TensptrT gl(derive(sum, leaf.get()));
	ade::TensptrT gr(derive(sum, leaf1.get()));

	std::stringstream lstr;
	std::stringstream rstr;

	lstr <<
		"(+)\n" <<
		"`--(*)\n" <<
		"    `--(+)\n" <<
		"    |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"    |   `--(+)\n" <<
		"    |       `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"    `--(+)\n" <<
		"        `--(*)\n" <<
		"            `--(*)\n" <<
		"            |   `--(*)\n" <<
		"            |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |   `--(+)\n" <<
		"            |       `--(*)\n" <<
		"            |           `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |           `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            `--(+)\n" <<
		"                `--(+)\n" <<
		"                    `--([2\\3\\1\\1\\1\\1\\1\\1])\n";
	rstr <<
		"(+)\n" <<
		"`--(*)\n" <<
		"    `--(+)\n" <<
		"    |   `--(+)\n" <<
		"    |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"    |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"    `--(+)\n" <<
		"        `--(*)\n" <<
		"            `--(*)\n" <<
		"            |   `--(*)\n" <<
		"            |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |   |   `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |   `--(+)\n" <<
		"            |       `--(*)\n" <<
		"            |           `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            |           `--([2\\3\\1\\1\\1\\1\\1\\1])\n" <<
		"            `--(+)\n" <<
		"                `--(+)\n" <<
		"                    `--([2\\3\\1\\1\\1\\1\\1\\1])\n";

	TREE_EQ(lstr, gl);
	TREE_EQ(rstr, gr);
}


#endif // DISABLE_GRADER_TEST
