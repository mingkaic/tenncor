
#include <future>

#include <boost/bimap.hpp>
#include <boost/lexical_cast.hpp>

#include "distrib/services/io/service.hpp"
#include "distrib/services/op/service.hpp"

#ifndef DISTRIB_MANAGER_HPP
#define DISTRIB_MANAGER_HPP

namespace distr
{

const std::string default_service = "tenncor";

struct iDistrManager
{
	virtual ~iDistrManager (void) = default;

	virtual std::string get_id (void) const = 0;

	virtual DistrIOService& get_io (void) = 0;

	virtual DistrOpService& get_op (void) = 0;
};

using iDistrMgrptrT = std::shared_ptr<iDistrManager>;

struct DistrManager final : public iDistrManager
{
	DistrManager (
		ppconsul::Consul& consul, size_t port,
		const std::string& svc_name = default_service,
		const std::string& id = "",
		const egrpc::ClientConfig& cfg = egrpc::ClientConfig(),
		size_t nthreads = 3)
	{
		std::string svc_id = id.empty() ?
			boost::uuids::to_string(global::get_uuidengine()()) : id;
		consul_ = std::make_unique<ConsulService>(
			consul, port, svc_id, svc_name);

		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		cq_ = builder.AddCompletionQueue();

		iosvc_ = std::make_unique<DistrIOService>(consul_.get(), cfg, builder);
		opsvc_ = std::make_unique<DistrOpService>(consul_.get(), cfg, builder, iosvc_.get());

		server_ = builder.BuildAndStart();
		global::infof("[server %s] listening on %s", svc_id.c_str(), address.c_str());

		iosvc_->initialize_server_call(*cq_);
		opsvc_->initialize_server_call(*cq_);

		for (size_t i = 0; i < nthreads; ++i)
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
		consul_(std::move(other.consul_)),
		iosvc_(std::move(other.iosvc_)),
		opsvc_(std::move(other.opsvc_)),
		cq_(std::move(cq_)),
		server_(std::move(server_)),
		rpc_jobs_(std::move(other.rpc_jobs_)) {}

	std::string get_id (void) const override
	{
		return consul_->id_;
	}

	DistrIOService& get_io (void) override
	{
		return *iosvc_;
	}

	DistrOpService& get_op (void) override
	{
		return *opsvc_;
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

	std::unique_ptr<ConsulService> consul_;

	std::unique_ptr<DistrIOService> iosvc_;

	std::unique_ptr<DistrOpService> opsvc_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	std::vector<std::thread> rpc_jobs_;
};

using DistrMgrptrT = std::shared_ptr<DistrManager>;

}

#endif // DISTRIB_MANAGER_HPP
