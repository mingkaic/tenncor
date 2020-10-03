
#ifndef DISTR_MOCK_MAKE_MGR_HPP
#define DISTR_MOCK_MAKE_MGR_HPP

#include "tenncor/distr/distr.hpp"

#include "tenncor/distr/mock/p2p.hpp"

struct DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (size_t port,
		const std::vector<distr::RegisterSvcF>& services,
		const std::string& alias = "")
	{
		std::string svc_id = alias.empty() ? global::get_generator()->get_str() : alias;
		peers_.emplace(svc_id, fmts::sprintf("0.0.0.0:%d", port));
		auto consul_svc = new MockP2P(kv_mtx_, svc_id, peers_, kv_);
		distr::PeerServiceConfig cfg(consul_svc, egrpc::ClientConfig(
			std::chrono::milliseconds(5000),
			std::chrono::milliseconds(10000),
			5
		));
		estd::ConfigMap<> svcs;
		for (auto& reg : services)
		{
			assert(nullptr == reg(svcs, cfg));
		}
		return std::make_shared<distr::DistrManager>(
			distr::P2PSvcptrT(consul_svc), svcs);
	}

	std::mutex kv_mtx_;

	types::StrUMapT<std::string> peers_;

	types::StrUMapT<std::string> kv_;
};

#endif // DISTR_MOCK_MAKE_MGR_HPP
