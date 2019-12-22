#define DISABLE_MATCHER_TEST
#ifndef DISABLE_MATCHER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/eteq.hpp"

#include "opt/voter.hpp"
#include "opt/matcher.hpp"


#define ELEMENTARY(LABEL, TYPE)opt::VoterArg{\
	LABEL,teq::ShaperT(),teq::ShaperT(), TYPE}


TEST(MATCHER, OrdrAny)
{
	std::vector<teq::DimT> slist = {3, 2};
	teq::Shape shape(slist);
	eteq::ETensor<float> a = eteq::make_variable_scalar<float>(2, shape);
	eteq::ETensor<float> b = eteq::make_variable_scalar<float>(3, shape);

	auto f1 = tenncor::pow(a, b);
	auto f2 = tenncor::pow(a, a);
	auto f3 = tenncor::pow(b, b);

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

	f2->accept(matcher);
	f1->accept(matcher);
	f3->accept(matcher);

	ASSERT_TRUE(estd::has(matcher.candidates_, a.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, b.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f1.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f2.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f3.get()));

	auto& acands = matcher.candidates_[a.get()]; // expect empty
	auto& bcands = matcher.candidates_[b.get()]; // expect empty

	EXPECT_EQ(0, acands.size());
	EXPECT_EQ(0, bcands.size());

	auto& f1cands = matcher.candidates_[f1.get()]; // expect {diff}
	auto& f2cands = matcher.candidates_[f2.get()]; // expect {same,diff}
	auto& f3cands = matcher.candidates_[f3.get()]; // expect {same,diff}

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
	std::vector<teq::DimT> slist = {3, 2};
	teq::Shape shape(slist);
	eteq::ETensor<float> a = eteq::make_variable_scalar<float>(2, shape);
	eteq::ETensor<float> b = eteq::make_variable_scalar<float>(3, shape);

	auto f1 = a * b;
	auto f2 = a * a;
	auto f3 = b * b;

	auto f4_sub_l = f1 * a; // match against similar
	auto f4_sub_r = f1 * b; // match against similar

	auto f5 = f2 * a; // match against everything except same
	auto f6 = f2 * b; // match against weird

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

	f2->accept(matcher);
	f1->accept(matcher);
	f3->accept(matcher);
	f4_sub_l->accept(matcher);
	f4_sub_r->accept(matcher);
	f5->accept(matcher);
	f6->accept(matcher);

	ASSERT_TRUE(estd::has(matcher.candidates_, a.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, b.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f1.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f2.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f3.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f4_sub_l.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f4_sub_r.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f5.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, f6.get()));

	auto& acands = matcher.candidates_[a.get()]; // expect empty
	auto& bcands = matcher.candidates_[b.get()]; // expect empty

	EXPECT_EQ(0, acands.size());
	EXPECT_EQ(0, bcands.size());

	auto& f1cands = matcher.candidates_[f1.get()]; // expect {diff}
	auto& f2cands = matcher.candidates_[f2.get()]; // expect {same,diff}
	auto& f3cands = matcher.candidates_[f3.get()]; // expect {same,diff}
	auto& f4sublcands = matcher.candidates_[f4_sub_l.get()]; // expect {similar,diff}
	auto& f4subrcands = matcher.candidates_[f4_sub_r.get()]; // expect {similar,diff}
	auto& f5cands = matcher.candidates_[f5.get()]; // expect {similar,diff,weird}
	auto& f6cands = matcher.candidates_[f6.get()]; // expect {weird,diff}

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
	std::vector<teq::DimT> slist = {3, 2};
	teq::Shape shape(slist);
	eteq::ETensor<float> a = eteq::make_variable_scalar<float>(2, shape);
	eteq::ETensor<float> b = eteq::make_variable_scalar<float>(3, shape);

	auto same = a * b;
	auto sub_l = same * a; // match against similar and similar2
	auto sub_r = same * b; // match against similar and similar2

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

	sub_ltens->accept(matcher);
	sub_rtens->accept(matcher);

	ASSERT_TRUE(estd::has(matcher.candidates_, a.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, b.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, sub_ltens.get()));
	ASSERT_TRUE(estd::has(matcher.candidates_, sub_rtens.get()));

	auto& acands = matcher.candidates_[a.get()]; // expect empty
	auto& bcands = matcher.candidates_[b.get()]; // expect empty

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
