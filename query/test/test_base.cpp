
#ifndef DISABLE_MATCHER_TEST


#include "gtest/gtest.h"

#include "dbg/print/teq.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "query/querier.hpp"
#include "query/parse.hpp"


TEST(BASE, NoComNoSymbs)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(2, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(SIN)\n"
		"_|___`--(constant:X)\n"
		"_`--(SUB)\n"
		"_____`--(constant:X)\n"
		"_____`--(constant:X)\n\n"
		"(SUB)\n"
		"_`--(constant:X)\n"
		"_`--(constant:X)\n";

	PrettyEquation peq;
	std::vector<std::string> dets;
	for (auto& det : roots)
	{
		std::stringstream ss;
		peq.print(ss, det);
		dets.push_back(ss.str());
	}
	std::sort(dets.begin(), dets.end());
	std::string got_det = fmts::join("\n", dets.begin(), dets.end());

	EXPECT_STREQ(expected, got_det.c_str());
}


TEST(BASE, NoComBadStruct)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"op\":{"
				"\"opname\":\"SUB\","
				"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
			"}},{\"symb\":\"C\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	ASSERT_EQ(0, detections.size());
}


TEST(BASE, NoComSymbOnly)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"A\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());
	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:X)\n"
		"_`--(constant:X)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(BASE, CommNoSymbs)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"ADD", 0})
	}, teq::Opcode{"ADD", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"ADD\","
			"\"args\":[{\"op\":{"
				"\"opname\":\"ADD\","
				"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
			"}},{\"symb\":\"C\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());
	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(ADD)\n"
		"_`--(SIN)\n"
		"_|___`--(constant:X)\n"
		"_`--(ADD)\n"
		"_____`--(constant:X)\n"
		"_____`--(constant:X)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(BASE, CommBadStruct)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"ADD", 0})
	}, teq::Opcode{"ADD", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"ADD\","
			"\"args\":[{\"op\":{"
				"\"opname\":\"SUB\","
				"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
			"}},{\"symb\":\"C\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	ASSERT_EQ(0, detections.size());
}


TEST(BASE, CommSymbOnly)
{
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{
		std::make_shared<MockFunctor>(teq::TensptrsT{x}, teq::Opcode{"SIN", 0}),
		std::make_shared<MockFunctor>(teq::TensptrsT{x,x}, teq::Opcode{"ADD", 0})
	}, teq::Opcode{"ADD", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"ADD\","
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"A\"}]"
		"}}";
	query::Node cond;
	json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());
	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(ADD)\n"
		"_`--(constant:X)\n"
		"_`--(constant:X)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


#endif // DISABLE_MATCHER_TEST
