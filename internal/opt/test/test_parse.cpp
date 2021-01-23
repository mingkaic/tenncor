
#ifndef DISABLE_OPT_PARSE_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/opt/mock/mock.hpp"


#ifdef CMAKE_SOURCE_DIR
const std::string testdir = std::string(CMAKE_SOURCE_DIR) + "models/test";
#else
const std::string testdir = "models/test";
#endif


TEST(PARSE, Parse)
{
	opt::OptRulesT rules;
	std::stringstream ss;
	ss <<
	"{\"conversions\":[{"
		"\"srcs\":[{"
			"\"op\":{\"opname\":\"BIG_CHUNGUS\",\"args\":["
				"{\"symb\":\"DINGUS\"},"
				"{\"symb\":\"ZURDING\"}"
			"]}"
		"},{"
			"\"leaf\":{\"label\":\"LORD_FRAYDO\"}"
		"}],"
		"\"dest\":{"
			"\"op\":{\"opname\":\"DIGGUS_BICKUS\",\"args\":["
				"{\"symb\":\"INCONT\"},"
				"{\"cst\":{\"value\":3,\"shape\":\"ZINGA\"}},"
				"{\"op\":{\"opname\":\"INENTIA\"}}"
			"]}"
		"}"
	"}]}";

	auto incont = std::make_shared<MockTarget>();
	auto zinga = std::make_shared<MockTarget>();
	auto inentia = std::make_shared<MockTarget>();
	auto diggus = std::make_shared<MockTarget>();

	MockTargetFactory mockfac;
	EXPECT_CALL(mockfac, make_symbol("INCONT")).Times(1).WillOnce(Return(incont));
	EXPECT_CALL(mockfac, make_scalar(3, "ZINGA")).Times(1).WillOnce(Return(zinga));
	EXPECT_CALL(mockfac, make_functor("INENTIA", _, opt::TargptrsT{})).Times(1).WillOnce(Return(inentia));
	EXPECT_CALL(mockfac, make_functor("DIGGUS_BICKUS", _, opt::TargptrsT{incont, zinga, inentia})).Times(1).WillOnce(Return(diggus));

	opt::json_parse(rules, ss, mockfac);

	// verify target 
	ASSERT_EQ(1, rules.size());
	opt::OptRule& rule = rules.front();
	auto troot = rule.target_;
	EXPECT_EQ(diggus, troot);

	// verify source
	// test srcs
	teq::TensptrT leaf = make_var(teq::Shape(), "LORD_FRAYDO");
	teq::TensptrT leaf2 = make_var(teq::Shape(), "DINGUS_FINGUS");
	teq::TensptrT leaf3 = make_var(teq::Shape(), "ZA_WARDO");
	teq::TensptrT leaf4 = make_var(teq::Shape(), "ZINGLING");
	teq::TensptrT func = make_fnc("BIG_CHUNGUS", 0, teq::TensptrsT{leaf2, leaf3});
	teq::TensptrT func2 = make_fnc("AL_ZAMBONI", 0, teq::TensptrsT{leaf, leaf4});

	query::Query q;
	func->accept(q);
	func2->accept(q);

	query::QResultsT results;
	for (auto& match_src : rule.match_srcs_)
	{
		auto res = q.match(match_src);
		results.insert(results.end(), res.begin(), res.end());
	}

	EXPECT_EQ(2, results.size());
	teq::TensMapT<query::QueryResult> resmap;
	for (const auto& result : results)
	{
		resmap.emplace(result.root_, result);
	}
	ASSERT_HAS(resmap, func.get());
	ASSERT_HAS(resmap, leaf.get());

	auto& fsymb = resmap.at(func.get()).symbs_;
	ASSERT_EQ(2, fsymb.size());
	ASSERT_EQ(0, resmap.at(leaf.get()).symbs_.size());

	ASSERT_HAS(fsymb, "DINGUS");
	ASSERT_HAS(fsymb, "ZURDING");
	EXPECT_EQ(leaf2.get(), fsymb.at("DINGUS"));
	EXPECT_EQ(leaf3.get(), fsymb.at("ZURDING"));
}


#endif // DISABLE_OPT_PARSE_TEST
