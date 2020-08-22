
#include "tenncor/distrib/imanager.hpp"

#ifndef DISTRIB_MANAGER_HPP
#define DISTRIB_MANAGER_HPP

namespace distr
{

using ConsulSvcptrT = std::unique_ptr<ConsulService>;

const std::string alias_publish_key = "published_alias_";

struct DistrManager final : public iDistrManager
{
	DistrManager (ConsulSvcptrT&& consul,
		const estd::ConfigMap<>& svcs, size_t nthreads = 3) :
		svcs_(svcs), consul_(std::move(consul))
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", consul_->port_);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		cq_ = builder.AddCompletionQueue();

		auto svc_keys = svcs_.get_keys();
		for (auto& skey : svc_keys)
		{
			static_cast<iPeerService*>(svcs_.get_obj(skey))->
				register_service(builder);
		}

		server_ = builder.BuildAndStart();
		global::infof("[server %s] listening on %s",
			consul_->id_.c_str(), address.c_str());

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

	DistrManager (DistrManager&& other) :
		svcs_(std::move(other.svcs_)),
		consul_(std::move(other.consul_)),
		cq_(std::move(other.cq_)),
		server_(std::move(other.server_)),
		rpc_jobs_(std::move(other.rpc_jobs_)) {}

	std::string get_id (void) const override
	{
		return consul_->id_;
	}

	iPeerService* get_service (const std::string& svc_key) override
	{
		auto svc = svcs_.get_obj(svc_key);
		return static_cast<iPeerService*>(svc);
	}

	void alias_node (const std::string& alias, const std::string& id) override
	{
		consul_->set_kv(alias_publish_key + alias, id);
	}

	std::string dealias_node (const std::string& alias) override
	{
		return consul_->get_kv(alias_publish_key + alias, "");
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

	ConsulSvcptrT consul_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	std::vector<std::thread> rpc_jobs_;
};

using DistrMgrptrT = std::shared_ptr<DistrManager>;

}

#endif // DISTRIB_MANAGER_HPP
