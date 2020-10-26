#include "tenncor/eteq/opsvc/mock/service.hpp"

#ifdef DISTR_OPSVC_MOCK_SERVICE_HPP

error::ErrptrT register_mock_opsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<distr::io::DistrIOService*>(
		svcs.get_obj(distr::io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<distr::op::DistrOpService>(distr::op::opsvc_key,
	[&]
	{
		return new distr::op::DistrOpService(
			std::make_unique<eigen::Device>(std::numeric_limits<size_t>::max()),
			std::make_unique<eteq::DerivativeFuncs>(), cfg, iosvc,
			std::make_shared<MockDistrOpCliBuilder>(),
			std::make_shared<MockOpService>());
	});
	return nullptr;
}

#endif
