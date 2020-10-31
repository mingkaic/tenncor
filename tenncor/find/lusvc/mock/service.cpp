#include "tenncor/find/lusvc/mock/service.hpp"

#ifdef DISTR_LUSVC_MOCK_SERVICE_HPP

struct MockDistrLuCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::lu::DistrLuCli(new MockLuStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCliCQT>();
	}
};

error::ErrptrT register_mock_lusvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<distr::io::DistrIOService*>(
		svcs.get_obj(distr::io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<distr::lu::DistrLuService>(distr::lu::lusvc_key,
	[&]
	{
		return new distr::lu::DistrLuService(cfg, iosvc,
			std::make_shared<MockDistrLuCliBuilder>(),
			std::make_shared<MockLuService>());
	});
	return nullptr;
}

#endif
