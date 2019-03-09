
#ifndef DISABLE_GOPT_TEST


#include "gtest/gtest.h"

#include "dbg/ade.hpp"
#include "dbg/ade_csv.hpp"

#include "ead/ead.hpp"

#include "subgraph_match/gopt.hpp"


TEST(GOPT, NormalizeGraph)
{
	opt::TransformsT const_normalize = {
		opt::Transform(
			"(ADD|MUL|MIN|MAX)\\(\\d*\\),,"
			"([\\w\\(\\)\\[\\]<>]+),"
			"(constant\\(\\d*\\)|scalar\\(\\d+(?:\\.\\d+)?\\))"
			"-> $1(),,$3,$2")
	};

	ade::Shape shape({4, 6});
	ead::NodeptrT<double> left = ead::convert_to_node<double>(
		ead::make_variable_scalar<double>(0, shape, "left"));
	ead::NodeptrT<double> right = ead::make_constant_scalar<double>(4, shape);

	auto f = age::add(left, right);

	opt::GraphOpt normalizer(const_normalize);
	f->get_tensor()->accept(normalizer);

	auto root = normalizer.token_roots_[opt::encode_tens(f->get_tensor().get())];
	ASSERT_NE(nullptr, root);
	std::string rep = root->encode(3);

	std::string variable_label = "variable(" +
		opt::encode_tens(left->get_tensor().get()) + ")";
	std::string expect_rep = "ADD(),,scalar(4)," + variable_label;
	EXPECT_STREQ(expect_rep.c_str(), rep.c_str());

	auto optimals = normalizer.apply_optimization<double>({f});
	ASSERT_EQ(1, optimals.size());
	auto opt_f = optimals[0];
	ASSERT_NE(nullptr, opt_f);

	std::stringstream ss;
	ss <<
		"(ADD[4\\6\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(4([4\\6\\1\\1\\1\\1\\1\\1]))\n" <<
		" `--(left([4\\6\\1\\1\\1\\1\\1\\1]))";
	auto compare_str = compare_graph(ss, opt_f->get_tensor());
	EXPECT_EQ(0, compare_str.size()) << compare_str;
}


TEST(GOPT, ContinuedNormalizedGraph)
{
	opt::TransformsT const_normalize = {
		opt::Transform(
			"(ADD|MUL|MIN|MAX)\\(\\d*\\),,"
			"([\\w\\(\\)\\[\\]<>]+),"
			"(constant\\(\\d*\\)|scalar\\(\\d+(?:\\.\\d+)?\\))"
			"-> $1(),,$3,$2 ->"),
		opt::Transform(
			"(ADD|MUL|MIN|MAX)\\(\\d*\\),,"
			"([\\w\\(\\)\\[\\]<>]+),\\1\\(\\d*\\),,"
			"(?:(.+),)?([\\w\\(\\)\\[\\]<>]+),([\\w\\(\\)\\[\\]<>]+)"
			"-> $1(),,$1(),$5,,$2,$4,$3")
	};

	ade::Shape shape({4, 6});
	ead::NodeptrT<double> left2 = ead::convert_to_node<double>(
		ead::make_variable_scalar<double>(2, shape, "left2"));
	ead::NodeptrT<double> right2 = ead::convert_to_node<double>(
		ead::make_variable_scalar<double>(1, shape, "right2"));

	auto left = age::add(left2, right2);
	auto right = ead::make_constant_scalar<double>(4, shape);

	auto f = age::add(left, right);

	opt::GraphOpt normalizer(const_normalize);
	f->get_tensor()->accept(normalizer);

	auto root = normalizer.token_roots_[opt::encode_tens(f->get_tensor().get())];
	ASSERT_NE(nullptr, root);
	std::string rep = root->encode(3);

	// rule 1:
	// first transforms to ADD(),,  scalar(4),ADD(),,  variable(left2),variable(right2)
	// rule 2:
	// then transforms to ADD(),,  ADD(),variable(right2),,  scalar(4),variable(left2)
	std::string left2_label = "variable(" +
		opt::encode_tens(left2->get_tensor().get()) + ")";
	std::string right2_label = "variable(" +
		opt::encode_tens(right2->get_tensor().get()) + ")";
	std::string expect_rep = fmts::sprintf("ADD(),,ADD(),%s,,scalar(4),%s",
		right2_label.c_str(), left2_label.c_str());
	EXPECT_STREQ(expect_rep.c_str(), rep.c_str());

	auto optimals = normalizer.apply_optimization<double>({f});
	ASSERT_EQ(1, optimals.size());
	auto opt_f = optimals[0];
	ASSERT_NE(nullptr, opt_f);

	std::stringstream ss;
	ss <<
		"(ADD[4\\6\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(ADD[4\\6\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(4([4\\6\\1\\1\\1\\1\\1\\1]))\n" <<
		" |   `--(left2([4\\6\\1\\1\\1\\1\\1\\1]))\n" <<
		" `--(right2([4\\6\\1\\1\\1\\1\\1\\1]))";
	auto compare_str = compare_graph(ss, opt_f->get_tensor());
	EXPECT_EQ(0, compare_str.size()) << compare_str;
}


#endif // DISABLE_GOPT_TEST
