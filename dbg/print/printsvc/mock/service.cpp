#include "dbg/print/printsvc/mock/service.hpp"

#ifdef DISTR_PRINTSVC_MOCK_SERVICE_HPP

struct MockDistrPrintCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::print::DistrPrintCli(new MockPrintStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCQueue>();
	}
};

error::ErrptrT register_mock_printsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<distr::io::DistrIOService*>(
		svcs.get_obj(distr::io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("printsvc requires iosvc already registered");
	}
	svcs.add_entry<distr::print::DistrPrintService>(distr::print::printsvc_key,
	[&]()
	{
		return new distr::print::DistrPrintService(cfg, iosvc, PrintEqConfig(),
			std::make_shared<MockDistrPrintCliBuilder>(),
			std::make_shared<MockPrintService>());
	});
	return nullptr;
}

#endif
