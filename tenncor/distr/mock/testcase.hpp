
#ifndef DISTR_MOCK_MAKE_MGR_HPP
#define DISTR_MOCK_MAKE_MGR_HPP

#include "tenncor/distr/distr.hpp"

#include "tenncor/distr/mock/p2p.hpp"

#include "testutil/tutil.hpp"

static const size_t nretries = 10;

struct DistrTestcase
{
	virtual ~DistrTestcase (void)
	{
		wait_for_ports();
	}

protected:
	distr::iDistrMgrptrT make_mgr (size_t port,
		const std::vector<distr::RegisterSvcF>& services,
		const std::string& alias = "")
	{
		std::string svc_id = alias.empty() ? global::get_generator()->get_str() : alias;
		assert(false == estd::has(peers_, svc_id));
		auto address = fmts::sprintf("0.0.0.0:%d", port);
		auto consul_svc = new MockP2P(kv_mtx_, svc_id, address, peers_, kv_);
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
		auto out = std::make_shared<distr::DistrManager<>>(
			distr::P2PSvcptrT(consul_svc), svcs);
		peers_.emplace(svc_id, address);
		return out;
	}

	[[nodiscard]]
	unsigned short reserve_port (unsigned short port = 5111)
	{
		while (estd::has(reserved_ports_, port) ||
			tutil::port_in_use(port))
		{
			++port;
		}
		reserved_ports_.emplace(port);
		return port;
	}

	void clean_up (void)
	{
		std::lock_guard<std::mutex> guard(kv_mtx_);
		peers_.clear();
		kv_.clear();
	}

	void wait_for_ports (void)
	{
		std::vector<std::thread> waits;
		for (auto port : reserved_ports_)
		{
			waits.push_back(std::thread(
			[port]()
			{
				for (size_t i = 0; i < nretries && tutil::port_in_use(port); ++i)
				{
					std::this_thread::sleep_for(std::chrono::milliseconds(50));
				}
				if (tutil::port_in_use(port))
				{
					global::errorf("port %d is still in use", port);
				}
			}));
		}
		for (auto& wait : waits)
		{
			wait.join();
		}
	}

	std::mutex kv_mtx_;

	types::StrUMapT<std::string> peers_;

	types::StrUMapT<std::string> kv_;

	std::unordered_set<unsigned short> reserved_ports_;
};

#endif // DISTR_MOCK_MAKE_MGR_HPP
