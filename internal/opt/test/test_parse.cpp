
#ifndef DISABLE_OPT_PARSE_TEST


#include <sstream>

#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/functor.hpp"

#include "internal/opt/opt.hpp"

#include "internal/opt/mock/target.hpp"


const std::string testdir = "models/test";


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
	opt::json_parse(rules, ss, MockTargetFactory());

	// test target
	ASSERT_EQ(1, rules.size());
	opt::OptRule& rule = rules.front();
	auto troot = dynamic_cast<MockTarget*>(rule.target_.get());
	ASSERT_NE(nullptr, troot);
	EXPECT_STREQ("DIGGUS_BICKUS:3", troot->tag_->to_string().c_str());
	ASSERT_EQ(3, troot->targs_.size());

	std::array<std::string,3> labels = {"INCONT:0", "3:ZINGA:1", "INENTIA:2"};
	for (size_t i = 0, n = troot->targs_.size(); i < n; ++i)
	{
		const auto& targ = troot->targs_.at(i);
		auto arg = dynamic_cast<MockTarget*>(targ.get());
		ASSERT_NE(nullptr, arg);
		EXPECT_STREQ(labels[i].c_str(), arg->tag_->to_string().c_str());
		EXPECT_EQ(0, arg->targs_.size());
	}

	// test srcs
	teq::TensptrT leaf = std::make_shared<MockLeaf>(teq::Shape(), "LORD_FRAYDO");
	teq::TensptrT leaf2 = std::make_shared<MockLeaf>(teq::Shape(), "DINGUS_FINGUS");
	teq::TensptrT leaf3 = std::make_shared<MockLeaf>(teq::Shape(), "ZA_WARDO");
	teq::TensptrT leaf4 = std::make_shared<MockLeaf>(teq::Shape(), "ZINGLING");
	teq::TensptrT func = std::make_shared<MockFunctor>(teq::TensptrsT{leaf2, leaf3}, teq::Opcode{"BIG_CHUNGUS", 0});
	teq::TensptrT func2 = std::make_shared<MockFunctor>(teq::TensptrsT{leaf, leaf4}, teq::Opcode{"AL_ZAMBONI", 0});

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
