
#ifndef DISTR_MANAGER_HPP
#define DISTR_MANAGER_HPP

#include "tenncor/distr/imanager.hpp"

namespace distr
{

struct DistrManager final : public iDistrManager
{
	DistrManager (P2PSvcptrT&& p2p,
		const estd::ConfigMap<>& svcs, size_t nthreads = 3,
		std::shared_ptr<iServerBuilder> builder =
			std::make_shared<ServerBuilder>()) :
		svcs_(svcs), p2p_(std::move(p2p)), cq_(builder->add_completion_queue())
	{
		std::string address = p2p_->get_local_addr();
		builder->add_listening_port(address,
			grpc::InsecureServerCredentials());

		auto svc_keys = svcs_.get_keys();
		for (auto& skey : svc_keys)
		{
			static_cast<iPeerService*>(svcs_.get_obj(skey))->
				register_service(*builder);
		}

		server_ = builder->build_and_start();
		global::infof("[server %s] listening on %s",
			p2p_->get_local_peer().c_str(), address.c_str());

		for (auto& skey : svc_keys)
		{
			static_cast<iPeerService*>(svcs_.get_obj(skey))->
				initialize_server_call(*cq_);
		}

		for (size_t i = 0, nlimits = nthreads > 0 ? nthreads : 1;
			i < nlimits; ++i)
		{
			rpc_jobs_.push_back(std::thread(
				&DistrManager::handle_rpcs, this));
		}
	}

	~DistrManager (void)
	{
		server_->shutdown();
		cq_->shutdown();
		for (auto& rpc_job : rpc_jobs_)
		{
			rpc_job.join();
		}
	}

	DistrManager (DistrManager&& other) = delete;

	DistrManager& operator = (DistrManager&& other) = delete;

	std::string get_id (void) const override
	{
		return p2p_->get_local_peer();
	}

	iPeerService* get_service (const std::string& svc_key) override
	{
		auto svc = svcs_.get_obj(svc_key);
		return static_cast<iPeerService*>(svc);
	}

	iP2PService* get_p2psvc (void)
	{
		return p2p_.get();
	}

private:
	// This can be run in multiple threads if needed.
	void handle_rpcs (void)
	{
		void* tag;
		bool ok = true;
		while (cq_->next(&tag, &ok))
		{
			auto call = static_cast<egrpc::iServerCall*>(tag);
			if (ok)
			{
				call->serve();
			}
			else
			{
				call->shutdown();
			}
		}
	}

	estd::ConfigMap<> svcs_;

	P2PSvcptrT p2p_;

	CQueueptrT cq_;

	std::unique_ptr<iServer> server_;

	std::vector<std::thread> rpc_jobs_;
};

void set_distrmgr (iDistrMgrptrT mgr,
	global::CfgMapptrT ctx = global::context());

iDistrManager* get_distrmgr (
	const global::CfgMapptrT& ctx = global::context());

}

#endif // DISTR_MANAGER_HPP
