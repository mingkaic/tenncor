
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

using iDistMgrptrT = std::shared_ptr<iDistrManager>;

struct DistrManager final : public iDistrManager
{
	DistrManager (
		ppconsul::Consul& consul, size_t port,
		const std::string& service = default_service,
		const std::string& id = "",
		const egrpc::ClientConfig& cfg = egrpc::ClientConfig(),
		size_t nthreads = 3) :
		consul_(consul, port, (id.empty() ?
			boost::uuids::to_string(eigen::rand_uuid_gen()()) :
			id), service)
	{
		std::string address = fmts::sprintf("0.0.0.0:%d", port);
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address,
			grpc::InsecureServerCredentials());
		cq_ = builder.AddCompletionQueue();

		iosvc_ = std::make_unique<DistrIOService>(consul_, cfg, builder);
		opsvc_ = std::make_unique<DistrOpService>(consul_, cfg, builder, iosvc_.get());

		server_ = builder.BuildAndStart();
		teq::infof("[server %s] listening on %s", consul_.id_.c_str(), address.c_str());

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

	std::string get_id (void) const override
	{
		return consul_.id_;
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

	ConsulService consul_;

	std::unique_ptr<DistrIOService> iosvc_;

	std::unique_ptr<DistrOpService> opsvc_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	std::vector<std::thread> rpc_jobs_;
};

using DistMgrptrT = std::shared_ptr<DistrManager>;

struct ManagerOwner final : public eigen::iOwner
{
	ManagerOwner (iDistMgrptrT mgr) : mgr_(mgr) {}

	void* get_raw (void) override
	{
		return mgr_.get();
	}

	iDistMgrptrT mgr_;
};

}

#endif // DISTRIB_MANAGER_HPP
