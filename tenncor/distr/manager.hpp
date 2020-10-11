
#ifndef DISTR_MANAGER_HPP
#define DISTR_MANAGER_HPP

#include "tenncor/distr/imanager.hpp"

namespace distr
{

struct DistrManager final : public iDistrManager
{
	DistrManager (P2PSvcptrT&& p2p,
		const estd::ConfigMap<>& svcs, size_t nthreads = 3) :
		svcs_(svcs), p2p_(std::move(p2p))
	{
		std::string address = p2p_->get_local_addr();
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		cq_ = builder.AddCompletionQueue();

		p2p_->register_service(builder);
		auto svc_keys = svcs_.get_keys();
		for (auto& skey : svc_keys)
		{
			static_cast<iPeerService*>(svcs_.get_obj(skey))->
				register_service(builder);
		}

		server_ = builder.BuildAndStart();
		global::infof("[server %s] listening on %s",
			p2p_->get_local_peer().c_str(), address.c_str());

		p2p_->initialize_server_call(*cq_);
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
		server_->Shutdown();
		cq_->Shutdown();
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
		while (cq_->Next(&tag, &ok))
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

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	std::vector<std::thread> rpc_jobs_;
};

using DistrMgrptrT = std::shared_ptr<DistrManager>;

void set_distrmgr (iDistrMgrptrT mgr,
	global::CfgMapptrT ctx = global::context());

iDistrManager* get_distrmgr (
	const global::CfgMapptrT& ctx = global::context());

}

#endif // DISTR_MANAGER_HPP
