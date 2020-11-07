#include "tenncor/distr/iosvc/mock/service.hpp"

#ifdef DISTR_IOSVC_MOCK_SERVICE_HPP

struct MockDistrIOCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::io::DistrIOCli(new MockIOStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCliCQT>();
	}
};

error::ErrptrT register_mock_iosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	svcs.add_entry<distr::io::DistrIOService>(distr::io::iosvc_key,
	[&]
	{
		return new distr::io::DistrIOService(cfg,
			std::make_shared<MockDistrIOCliBuilder>(),
			std::make_shared<MockIOService>());
	});
	return nullptr;
}

#endif
