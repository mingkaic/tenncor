
#ifndef DISABLE_QUERY_BASE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "dbg/print/teq.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/query/querier.hpp"
#include "internal/query/parse.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


TEST(BASE, BadParse)
{
	auto* logger = new exam::MockLogger();
	global::set_logger(logger);

	std::stringstream badjson;
	query::Node cond;
	std::string fatalmsg = "failed to parse json condition";
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(query::json_parse(cond, badjson), fatalmsg.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(BASE, ErasedNode)
{
	auto c1 = std::make_shared<MockLeaf>(teq::Shape(), "3.2");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto interm = std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0});
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2, interm}, teq::Opcode{"SUB", 0});
	query::Query matcher;
	root->accept(matcher);

	matcher.erase(interm.get());
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
	query::json_parse(cond, condjson);
	auto detections = matcher.match(cond);
	EXPECT_EQ(0, detections.size());

	std::stringstream condjson2;
	condjson2 <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"symb\":\"A\"},{\"symb\":\"B\"}]"
		"}}";
	query::Node cond2;
	query::json_parse(cond2, condjson2);
	auto detections2 = matcher.match(cond2);
	ASSERT_EQ(1, detections2.size());
	teq::iTensor* res = detections2.front();
	EXPECT_EQ(root.get(), res);
}


TEST(BASE, BadNode)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);

	auto c1 = std::make_shared<MockLeaf>(teq::Shape(), "3.2");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2,
		std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});
	query::Query matcher;
	root->accept(matcher);

	std::stringstream badjson;
	badjson << "{}";
	{
		query::Node cond;
		query::json_parse(cond, badjson);
		std::string fatalmsg = "cannot look for unknown node";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(matcher.match(cond), fatalmsg.c_str());
	}

	std::stringstream badjson2;
	badjson2 <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"symb\":\"A\"},{}]"
		"}}";
	{
		query::Node cond;
		query::json_parse(cond, badjson2);
		std::string fatalmsg = "cannot look for unknown node";
		EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillOnce(Return(true));
		EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
		EXPECT_FATAL(matcher.match(cond), fatalmsg.c_str());
	}

	global::set_logger(new exam::NoSupportLogger());
}


TEST(BASE, DirectConstants)
{
	auto c1 = std::make_shared<MockLeaf>(teq::Shape(), "3.2");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2,
		std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson << "{\"cst\":3.5}";
	query::Node cond;
	query::json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	EXPECT_EQ(c2.get(), *roots.begin());
}


TEST(BASE, Constants)
{
	auto c1 = std::make_shared<MockLeaf>(teq::Shape(), "3.2");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "3.5");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2,
		std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":[{\"cst\":3.5},{\"symb\":\"A\"}]"
		"}}";
	query::Node cond;
	query::json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	char expected[] =
		"(SUB)\n"
		"_`--(constant:3.5)\n"
		"_`--(SUB)\n"
		"_____`--(constant:3.2)\n"
		"_____`--(constant:X)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, *roots.begin());
	EXPECT_STREQ(expected, ss.str().c_str());
}


TEST(BASE, DirectLeafs)
{
	auto c1 = std::make_shared<MockLeaf>(teq::Shape({1, 2}), "X");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto x = std::make_shared<MockLeaf>(teq::Shape({1, 2}), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2,
		std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});
	x->meta_.tname_ = "specops";

	std::stringstream condjson;
	condjson << "{\"leaf\":{"
		"\"label\":\"X\","
		"\"shape\":[1,2],"
		"\"dtype\":\"specops\""
	"}}";
	query::Node cond;
	query::json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);
	teq::TensSetT roots(detections.begin(), detections.end());

	ASSERT_EQ(1, roots.size());
	EXPECT_EQ(x.get(), *roots.begin());
}


TEST(BASE, Leafs)
{
	auto c1 = std::make_shared<MockLeaf>(teq::Shape({1, 2}), "X");
	auto c2 = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto x = std::make_shared<MockLeaf>(teq::Shape(), "X");
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{c2,
		std::make_shared<MockFunctor>(teq::TensptrsT{c1,x}, teq::Opcode{"SUB", 0})
	}, teq::Opcode{"SUB", 0});

	std::stringstream condjson;
	condjson <<
		"{\"op\":{"
			"\"opname\":\"SUB\","
			"\"args\":["
			"{"
				"\"leaf\":{"
					"\"label\":\"X\","
					"\"shape\":[1,2]"
				"}"
			"},{\"symb\":\"A\"}]"
		"}}";
	query::Node cond;
	query::json_parse(cond, condjson);
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
	query::json_parse(cond, condjson);
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
	types::StringsT dets;
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
	query::json_parse(cond, condjson);
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
	query::json_parse(cond, condjson);
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
	query::json_parse(cond, condjson);
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
	query::json_parse(cond, condjson);
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
	query::json_parse(cond, condjson);
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


TEST(BASE, Capture)
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
			"\"args\":[{\"symb\":\"A\"},"
				"{\"op\":{"
					"\"opname\":\"SUB\","
					"\"capture\":\"stuff\","
					"\"args\":[{\"symb\":\"B\"},{\"symb\":\"C\"}]"
				"}}]"
		"}}";
	query::Node cond;
	query::json_parse(cond, condjson);
	query::Query matcher;
	root->accept(matcher);
	auto detections = matcher.match(cond);

	ASSERT_EQ(1, detections.size());
	ASSERT_HAS(detections.front().symbs_, "stuff");
	char expected[] =
		"(SUB)\n"
		"_`--(constant:X)\n"
		"_`--(constant:X)\n";

	PrettyEquation peq;
	std::stringstream ss;
	peq.print(ss, detections.front().symbs_["stuff"]);
	EXPECT_STREQ(expected, ss.str().c_str());
}


#endif // DISABLE_QUERY_BASE_TEST
