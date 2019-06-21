#include "ead/ead.hpp"

#include "testutil/common.hpp"

#include "gtest/gtest.h"

#include "experimental/match/matcher.hpp"


TEST(MATCH, OrdrAny)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
    ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

    auto f1 = age::pow(a, b);
    auto f2 = age::pow(a, a);
    auto f3 = age::pow(b, b);

    opt::match::VoterPool vpool;
    {
        auto voter = std::make_unique<opt::match::OrdrVoter>("POW");
        voter->emplace({
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            },
            opt::match::VoterArg{
                "B",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::INTERM, "diff"});
        voter->emplace({
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            },
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::INTERM, "same"});
        vpool.branches_.emplace(voter->label_, std::move(voter));
    }
    opt::match::Matcher matcher(vpool);

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
    EXPECT_TRUE(util::has(f1cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f2cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f3cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f2cands, opt::match::Symbol{opt::match::INTERM, "same"}));
    EXPECT_TRUE(util::has(f3cands, opt::match::Symbol{opt::match::INTERM, "same"}));
}


TEST(MATCH, CommAny)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
    ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

    auto f1 = age::mul(a, b);
    auto f2 = age::mul(a, a);
    auto f3 = age::mul(b, b);

    auto f4_sub_l = age::mul(f1, a); // match against similar and similar2
    auto f4_sub_r = age::mul(f1, b); // match against similar and similar2

    auto f5 = age::mul(f2, a); // match against everything except same
    auto f6 = age::mul(f2, b); // match against weird

    opt::match::VoterPool vpool;
    {
        auto voter = std::make_unique<opt::match::CommVoter>("MUL");
        voter->emplace({
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            },
            opt::match::VoterArg{
                "B",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::INTERM, "diff"});
        voter->emplace({
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            },
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::INTERM, "same"});
        voter->emplace({
            opt::match::VoterArg{
                "diff",
                ade::CoordptrT(),
                ade::CoordptrT(),
                BRANCH,
            },
            opt::match::VoterArg{
                "A",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::SCALAR, "similar"});
        voter->emplace({
            opt::match::VoterArg{
                "diff",
                ade::CoordptrT(),
                ade::CoordptrT(),
                BRANCH,
            },
            opt::match::VoterArg{
                "B",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::SCALAR, "similar2"});
        voter->emplace({
            opt::match::VoterArg{
                "same",
                ade::CoordptrT(),
                ade::CoordptrT(),
                BRANCH,
            },
            opt::match::VoterArg{
                "B",
                ade::CoordptrT(),
                ade::CoordptrT(),
                ANY,
            }
        }, opt::match::Symbol{opt::match::SCALAR, "weird"});
        vpool.branches_.emplace(voter->label_, std::move(voter));
    }
    opt::match::Matcher matcher(vpool);

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
    auto& f4sublcands = matcher.candidates_[f4_sub_ltens.get()]; // expect {similar,similar2,diff}
    auto& f4subrcands = matcher.candidates_[f4_sub_rtens.get()]; // expect {similar,similar2,diff}
    auto& f5cands = matcher.candidates_[f5tens.get()]; // expect {}
    auto& f6cands = matcher.candidates_[f6tens.get()]; // expect {weird}

    EXPECT_EQ(1, f1cands.size());
    EXPECT_EQ(2, f2cands.size());
    EXPECT_EQ(2, f3cands.size());
    EXPECT_EQ(2, f4sublcands.size());
    EXPECT_EQ(2, f4subrcands.size());
    EXPECT_EQ(4, f5cands.size());
    EXPECT_EQ(2, f6cands.size());

    EXPECT_TRUE(util::has(f1cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f2cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f3cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f2cands, opt::match::Symbol{opt::match::INTERM, "same"}));
    EXPECT_TRUE(util::has(f3cands, opt::match::Symbol{opt::match::INTERM, "same"}));
    EXPECT_TRUE(util::has(f4sublcands, opt::match::Symbol{opt::match::SCALAR, "similar"}));
    EXPECT_TRUE(util::has(f5cands, opt::match::Symbol{opt::match::SCALAR, "similar"}));
    EXPECT_TRUE(util::has(f4subrcands, opt::match::Symbol{opt::match::SCALAR, "similar2"}));
    EXPECT_TRUE(util::has(f5cands, opt::match::Symbol{opt::match::SCALAR, "similar2"}));
    EXPECT_TRUE(util::has(f4sublcands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f4subrcands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f5cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f6cands, opt::match::Symbol{opt::match::INTERM, "diff"}));
    EXPECT_TRUE(util::has(f5cands, opt::match::Symbol{opt::match::SCALAR, "weird"}));
    EXPECT_TRUE(util::has(f6cands, opt::match::Symbol{opt::match::SCALAR, "weird"}));
}


int main (int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
