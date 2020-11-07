#include "tenncor/serial/oxsvc/mock/service.hpp"

#ifdef DISTR_OXSVC_MOCK_SERVICE_HPP

struct MockDistrOxCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::ox::DistrSerializeCli(new MockOxStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCliCQT>();
	}
};

error::ErrptrT register_mock_oxsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<distr::io::DistrIOService*>(
		svcs.get_obj(distr::io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("oxsvc requires iosvc already registered");
	}
	svcs.add_entry<distr::ox::DistrSerializeService>(distr::ox::oxsvc_key,
	[&]
	{
		return new distr::ox::DistrSerializeService(cfg, iosvc,
			std::make_shared<MockDistrOxCliBuilder>(),
			std::make_shared<MockOxService>());
	});
	return nullptr;
}

#endif
