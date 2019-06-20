
struct MockCandidate final : public iCandidate
{
    //
};

TEST(VOTE, Any)
{
	std::vector<ade::DimT> slist = {3, 2};
	ade::Shape shape(slist);
    ead::NodeptrT<float> a = ead::make_variable_scalar<float>(2, shape);
	ead::NodeptrT<float> b = ead::make_variable_scalar<float>(3, shape);

    auto f1 = age::pow(a, b);
    auto f2 = age::pow(a, a);
    auto f3 = age::pow(b, b);

    VoterPool vpool;
    {
        auto voter = std::make_unique<OrdrVoter>;
        voter->label_ = "POW";

        auto diff_cand = std::make_shared<MockCandidate>();
        auto same_cand = std::make_shared<MockCandidate>();
        voter->emplace({
            VoterArg{
                "A",
                ade::CoordptrT(),
                CoordptrT(),
                ANY,
            },
            VoterArg{
                "B",
                ade::CoordptrT(),
                CoordptrT(),
                ANY,
            }
        }, diff_cand);
        voter->emplace({
            VoterArg{
                "A",
                ade::CoordptrT(),
                CoordptrT(),
                ANY,
            },
            VoterArg{
                "A",
                ade::CoordptrT(),
                CoordptrT(),
                ANY,
            }
        }, same_cand);
        vpool.branches_.emplace("POW", std::move(voter));
    }
    Matcher matcher(&vpool);

    f2->accept(matcher);
    f1->accept(matcher);
    f3->accept(matcher);

    auto acands = matcher[a->get_tensor().get()]; // expect empty
    auto bcands = matcher[b->get_tensor().get()]; // expect empty

    auto f1cands = matcher[f1->get_tensor().get()]; // expect {diff_cand}
    auto f2cands = matcher[f2->get_tensor().get()]; // expect {same_cand}
    auto f3cands = matcher[f3->get_tensor().get()]; // expect {same_cand}
}
