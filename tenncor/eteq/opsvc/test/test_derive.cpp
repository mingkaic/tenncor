
#ifndef DISABLE_OPSVC_DERIVE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/mock/mock.hpp"
#include "tenncor/distr/iosvc/mock/mock.hpp"
#include "tenncor/eteq/opsvc/mock/mock.hpp"


const std::string test_service = "tenncor.eteq.opsvc.test";


struct MockDeriveFunc final : public teq::iDerivativeFuncs
{
	static std::shared_ptr<MockLeaf> other_leaf;

	teq::TensptrT lderive (teq::FuncptrT op,
		teq::TensptrT supgrad, size_t arg_idx) const override
	{
		std::string label = op->to_string();
		teq::TensptrT local_der;
		if (label == "FUNC")
		{
			local_der = op->get_args()[arg_idx];
		}
		else if (label == "FUNC2")
		{
			local_der = std::make_shared<MockFunctor>(
				teq::TensptrsT{op->get_args()[arg_idx]}, teq::Opcode{"FUNC4", 3});
		}
		else
		{
			local_der = other_leaf;
		}
		teq::TensptrT tens(new MockFunctor(
			teq::TensptrsT{op,local_der}, teq::Opcode{"FUNC2", 1}));
		auto out = std::make_shared<MockFunctor>(
			teq::TensptrsT{tens, supgrad}, teq::Opcode{"FUNC3", 2});
		out->meta_.tcode_ = egen::DOUBLE;
		out->meta_.tname_ = "DOUBLE";
		return out;
	}

	teq::TensptrT get_const_one (teq::Shape shape) const override
	{
		other_leaf->meta_.tcode_ = egen::DOUBLE;
		other_leaf->meta_.tname_ = "DOUBLE";
		return other_leaf;
	}

	teq::TensptrT get_const_zero (teq::Shape shape) const override
	{
		other_leaf->meta_.tcode_ = egen::DOUBLE;
		other_leaf->meta_.tname_ = "DOUBLE";
		return other_leaf;
	}

	teq::TensptrT add (teq::TensptrsT elems) const override
	{
		auto out = std::make_shared<MockFunctor>(elems, teq::Opcode{"FUNC", 0});
		out->meta_.tcode_ = egen::DOUBLE;
		out->meta_.tname_ = "DOUBLE";
		return out;
	}
};


std::shared_ptr<MockLeaf> MockDeriveFunc::other_leaf =
	std::make_shared<MockLeaf>(teq::Shape({2, 2}), "other");


struct DERIVE : public ::testing::Test, public DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (const std::string& id)
	{
		return make_mgr(id, reserve_port());
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id, size_t port)
	{
		return DistrTestcase::make_local_mgr(port, {
			register_mock_iosvc,
			[](estd::ConfigMap<>& svcs, const distr::PeerServiceConfig& cfg) -> error::ErrptrT
			{
				auto iosvc = static_cast<distr::io::DistrIOService*>(svcs.get_obj(distr::io::iosvc_key));
				if (nullptr == iosvc)
				{
					return error::error("opsvc requires iosvc already registered");
				}
				svcs.add_entry<distr::op::DistrOpService>(distr::op::opsvc_key,
				[&]()
				{
					return new distr::op::DistrOpService(
						std::make_unique<eigen::Device>(std::numeric_limits<size_t>::max()),
						std::make_unique<MockDeriveFunc>(), cfg, iosvc,
						std::make_shared<MockDistrOpCliBuilder>(),
						std::make_shared<MockOpService>());
				});
				return nullptr;
			},
		}, id);
	}
};


TEST_F(DERIVE, RemoteDerivation)
{
	MockDeriveFunc builder;

	teq::Shape shape({2, 2});
	std::vector<double> data = {1, 2, 3, 4};

	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	auto a = std::make_shared<MockLeaf>(data, shape, "a");
	auto b = std::make_shared<MockLeaf>(data, shape, "b");
	auto c = std::make_shared<MockLeaf>(data, shape, "c");
	a->meta_.tcode_ = egen::DOUBLE;
	b->meta_.tcode_ = egen::DOUBLE;
	c->meta_.tcode_ = egen::DOUBLE;
	a->meta_.tname_ = "DOUBLE";
	b->meta_.tname_ = "DOUBLE";
	c->meta_.tname_ = "DOUBLE";

	auto d = std::make_shared<MockFunctor>(teq::TensptrsT{b, a}, data, teq::Opcode{"FUNC", 0});
	auto e = std::make_shared<MockFunctor>(teq::TensptrsT{c, d}, data, teq::Opcode{"FUNC", 0});
	auto farg = std::make_shared<MockFunctor>(teq::TensptrsT{d}, data, teq::Opcode{"FUNC2", 0});
	auto farg2 = std::make_shared<MockFunctor>(teq::TensptrsT{c}, data, teq::Opcode{"FUNC2", 0});
	auto f = std::make_shared<MockFunctor>(teq::TensptrsT{farg, farg2}, data, teq::Opcode{"FUNC", 0});
	d->meta_.tcode_ = egen::DOUBLE;
	e->meta_.tcode_ = egen::DOUBLE;
	farg->meta_.tcode_ = egen::DOUBLE;
	farg2->meta_.tcode_ = egen::DOUBLE;
	f->meta_.tcode_ = egen::DOUBLE;
	d->meta_.tname_ = "DOUBLE";
	e->meta_.tname_ = "DOUBLE";
	farg->meta_.tname_ = "DOUBLE";
	farg2->meta_.tname_ = "DOUBLE";
	f->meta_.tname_ = "DOUBLE";

	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{e, f}, data, teq::Opcode{"FUNC", 0});
	root->meta_.tcode_ = egen::DOUBLE;
	root->meta_.tname_ = "DOUBLE";

	auto& svc = distr::get_iosvc(*mgr);
	std::string root_id = svc.expose_node(root);
	std::string base_id = svc.expose_node(a);

	// instance 2
	distr::iDistrMgrptrT mgr2 = make_mgr("mgr2");
	auto& svc2 = distr::get_iosvc(*mgr2);

	error::ErrptrT err = nullptr;
	teq::TensptrT root_ref = svc2.lookup_node(err, root_id);
	ASSERT_NOERR(err);
	teq::TensptrT base_ref = svc2.lookup_node(err, base_id);
	ASSERT_NOERR(err);

	teq::TensMapT<teq::TensptrsT> grads = {
		{root_ref.get(), {builder.get_const_one(root->shape())}}
	};
	auto remote_ders = distr::get_opsvc(*mgr2).derive(
		grads, {root_ref}, distr::op::BackpropMeta{
			teq::TensSetT{base_ref.get()}});
	ASSERT_EQ(1, remote_ders.size());
	EXPECT_HAS(remote_ders, base_ref.get());
	auto dbase = remote_ders[base_ref.get()];

	auto dbaseref = dynamic_cast<distr::iDistrRef*>(dbase.get());
	ASSERT_NE(nullptr, dbaseref);
	EXPECT_EQ(mgr->get_id(), dbaseref->cluster_id());

	teq::TensptrT remoteder = svc.lookup_node(err, dbaseref->node_id());
	ASSERT_NOERR(err);

	auto other_id = svc2.lookup_id(MockDeriveFunc::other_leaf.get());
	ASSERT_TRUE(bool(other_id));
	std::string other = *other_id;

	EXPECT_GRAPHEQ(fmts::sprintf(
		"(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(FUNC4<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|_______`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___________`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___________`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|___`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|___|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|___|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|___`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___|_______`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|_______`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___________`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______|___________`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___|___`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|_______|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|_______|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|_______|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___|___________`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|_______|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|_______|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|_______|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________|___________`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___________`--(placeholder:mgr2/%s<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___|___`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(FUNC3<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(FUNC2<no_type>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___|___`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|_______|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|_______|_______`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|_______|_______`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|_______`--(FUNC2<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___|___________`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(constant:c<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|_______`--(FUNC<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(constant:b<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___________`--(constant:a<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(placeholder:mgr2/%s<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n",
		other.c_str(), other.c_str()),
		remoteder);
}


#endif // DISABLE_OPSVC_DERIVE_TEST
