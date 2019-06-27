
#ifndef DISABLE_MATCHER_TEST


#include "gtest/gtest.h"

#include "testutil/common.hpp"

#include "ead/ead.hpp"

#include "opt/voter.hpp"
#include "opt/matcher.hpp"


#define ELEMENTARY(LABEL, TYPE)opt::VoterArg{\
	LABEL,ade::CoordptrT(),ade::CoordptrT(), TYPE}


TEST(MATCHER, OrdrAny)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
	ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

	auto f1 = age::pow(a, b);
	auto f2 = age::pow(a, a);
	auto f3 = age::pow(b, b);

	opt::Matcher matcher;
	{
		opt::VoterPool& vpool = matcher.voters_;
		auto voter = std::make_shared<opt::OrdrVoter>("POW");
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::INTERM, "diff"});
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("A", ANY)},
			opt::Symbol{opt::INTERM, "same"});
		vpool.branches_.emplace(voter->label_, voter);
	}

	auto atens = a->get_tensor();
	auto btens = b->get_tensor();
	auto f1tens = f1->get_tensor();
	auto f2tens = f2->get_tensor();
	auto f3tens = f3->get_tensor();

	f2tens->accept(matcher);
	f1tens->accept(matcher);
	f3tens->accept(matcher);

	ASSERT_TRUE(util::has(matcher.candidates_, atens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, btens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f1tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f2tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f3tens.get()));

	auto& acands = matcher.candidates_[atens.get()]; // expect empty
	auto& bcands = matcher.candidates_[btens.get()]; // expect empty

	EXPECT_EQ(0, acands.size());
	EXPECT_EQ(0, bcands.size());

	auto& f1cands = matcher.candidates_[f1tens.get()]; // expect {diff}
	auto& f2cands = matcher.candidates_[f2tens.get()]; // expect {same,diff}
	auto& f3cands = matcher.candidates_[f3tens.get()]; // expect {same,diff}

	EXPECT_EQ(1, f1cands.size());
	EXPECT_EQ(2, f2cands.size());
	EXPECT_EQ(2, f3cands.size());
	EXPECT_HAS(f1cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f2cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f3cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f2cands, (opt::Symbol{opt::INTERM, "same"}));
	EXPECT_HAS(f3cands, (opt::Symbol{opt::INTERM, "same"}));
}


TEST(MATCHER, CommAny)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
	ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

	auto f1 = age::mul(a, b);
	auto f2 = age::mul(a, a);
	auto f3 = age::mul(b, b);

	auto f4_sub_l = age::mul(f1, a); // match against similar
	auto f4_sub_r = age::mul(f1, b); // match against similar

	auto f5 = age::mul(f2, a); // match against everything except same
	auto f6 = age::mul(f2, b); // match against weird

	opt::Matcher matcher;
	{
		opt::VoterPool& vpool = matcher.voters_;
		auto voter = std::make_shared<opt::CommVoter>("MUL");
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::INTERM, "diff"});
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("A", ANY)},
			opt::Symbol{opt::INTERM, "same"});
		voter->emplace({ELEMENTARY("diff", BRANCH), ELEMENTARY("A", ANY)},
			opt::Symbol{opt::SCALAR, "similar"});
		voter->emplace({ELEMENTARY("same", BRANCH), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::SCALAR, "weird"});
		vpool.branches_.emplace(voter->label_, voter);
	}

	auto atens = a->get_tensor();
	auto btens = b->get_tensor();
	auto f1tens = f1->get_tensor();
	auto f2tens = f2->get_tensor();
	auto f3tens = f3->get_tensor();
	auto f4_sub_ltens = f4_sub_l->get_tensor();
	auto f4_sub_rtens = f4_sub_r->get_tensor();
	auto f5tens = f5->get_tensor();
	auto f6tens = f6->get_tensor();

	f2tens->accept(matcher);
	f1tens->accept(matcher);
	f3tens->accept(matcher);
	f4_sub_ltens->accept(matcher);
	f4_sub_rtens->accept(matcher);
	f5tens->accept(matcher);
	f6tens->accept(matcher);

	ASSERT_TRUE(util::has(matcher.candidates_, atens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, btens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f1tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f2tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f3tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f4_sub_ltens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f4_sub_rtens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f5tens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, f6tens.get()));

	auto& acands = matcher.candidates_[atens.get()]; // expect empty
	auto& bcands = matcher.candidates_[btens.get()]; // expect empty

	EXPECT_EQ(0, acands.size());
	EXPECT_EQ(0, bcands.size());

	auto& f1cands = matcher.candidates_[f1tens.get()]; // expect {diff}
	auto& f2cands = matcher.candidates_[f2tens.get()]; // expect {same,diff}
	auto& f3cands = matcher.candidates_[f3tens.get()]; // expect {same,diff}
	auto& f4sublcands = matcher.candidates_[f4_sub_ltens.get()]; // expect {similar,diff}
	auto& f4subrcands = matcher.candidates_[f4_sub_rtens.get()]; // expect {similar,diff}
	auto& f5cands = matcher.candidates_[f5tens.get()]; // expect {similar,diff,weird}
	auto& f6cands = matcher.candidates_[f6tens.get()]; // expect {weird,diff}

	EXPECT_EQ(1, f1cands.size());
	EXPECT_EQ(2, f2cands.size());
	EXPECT_EQ(2, f3cands.size());
	EXPECT_EQ(2, f4sublcands.size());
	EXPECT_EQ(2, f4subrcands.size());
	EXPECT_EQ(3, f5cands.size());
	EXPECT_EQ(2, f6cands.size());

	EXPECT_HAS(f1cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f2cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f3cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f4sublcands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f4subrcands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f5cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f6cands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(f2cands, (opt::Symbol{opt::INTERM, "same"}));
	EXPECT_HAS(f3cands, (opt::Symbol{opt::INTERM, "same"}));
	EXPECT_HAS(f4sublcands, (opt::Symbol{opt::SCALAR, "similar"}));
	EXPECT_HAS(f4subrcands, (opt::Symbol{opt::SCALAR, "similar"}));
	EXPECT_HAS(f5cands, (opt::Symbol{opt::SCALAR, "similar"}));
	EXPECT_HAS(f5cands, (opt::Symbol{opt::SCALAR, "weird"}));
	EXPECT_HAS(f6cands, (opt::Symbol{opt::SCALAR, "weird"}));
}


TEST(MATCHER, Ambiguous_CommAny)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
	ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

	auto same = age::mul(a, b);
	auto sub_l = age::mul(same, a); // match against similar and similar2
	auto sub_r = age::mul(same, b); // match against similar and similar2

	opt::Matcher matcher;
	{
		opt::VoterPool& vpool = matcher.voters_;
		auto voter = std::make_shared<opt::CommVoter>("MUL");
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::INTERM, "diff"});
		voter->emplace({ELEMENTARY("A", ANY), ELEMENTARY("A", ANY)},
			opt::Symbol{opt::INTERM, "same"}); // expect to miss all
		voter->emplace({ELEMENTARY("diff", BRANCH), ELEMENTARY("A", ANY)},
			opt::Symbol{opt::SCALAR, "similar"});
		voter->emplace({ELEMENTARY("diff", BRANCH), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::SCALAR, "similar2"});
		voter->emplace({ELEMENTARY("same", BRANCH), ELEMENTARY("B", ANY)},
			opt::Symbol{opt::SCALAR, "weird"}); // expect to miss all
		vpool.branches_.emplace(voter->label_, voter);
	}

	auto atens = a->get_tensor();
	auto btens = b->get_tensor();
	auto sub_ltens = sub_l->get_tensor();
	auto sub_rtens = sub_r->get_tensor();

	sub_ltens->accept(matcher);
	sub_rtens->accept(matcher);

	ASSERT_TRUE(util::has(matcher.candidates_, atens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, btens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, sub_ltens.get()));
	ASSERT_TRUE(util::has(matcher.candidates_, sub_rtens.get()));

	auto& acands = matcher.candidates_[atens.get()]; // expect empty
	auto& bcands = matcher.candidates_[btens.get()]; // expect empty

	EXPECT_EQ(0, acands.size());
	EXPECT_EQ(0, bcands.size());

	auto& sublcands = matcher.candidates_[sub_ltens.get()]; // expect {similar,similar2,diff}
	auto& subrcands = matcher.candidates_[sub_rtens.get()]; // expect {similar,similar2,diff}

	EXPECT_EQ(3, sublcands.size());
	EXPECT_EQ(3, subrcands.size());

	EXPECT_HAS(sublcands, (opt::Symbol{opt::SCALAR, "similar"}));
	EXPECT_HAS(subrcands, (opt::Symbol{opt::SCALAR, "similar"}));
	EXPECT_HAS(sublcands, (opt::Symbol{opt::SCALAR, "similar2"}));
	EXPECT_HAS(subrcands, (opt::Symbol{opt::SCALAR, "similar2"}));
	EXPECT_HAS(sublcands, (opt::Symbol{opt::INTERM, "diff"}));
	EXPECT_HAS(subrcands, (opt::Symbol{opt::INTERM, "diff"}));
}


#endif // DISABLE_MATCHER_TEST
