#include "tenncor/hone/hosvc/mock/service.hpp"

#ifdef DISTR_HOSVC_MOCK_SERVICE_HPP

struct MockDistrHoCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::ho::DistrHoCli(new MockHoStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCQueue>();
	}
};

error::ErrptrT register_mock_hosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<distr::io::DistrIOService*>(
		svcs.get_obj(distr::io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("hosvc requires iosvc already registered");
	}
	svcs.add_entry<distr::ho::DistrHoService>(distr::ho::hosvc_key,
	[&]()
	{
		return new distr::ho::DistrHoService(cfg, iosvc,
			std::make_shared<MockDistrHoCliBuilder>(),
			std::make_shared<MockHoService>());
	});
	return nullptr;
}

#endif
