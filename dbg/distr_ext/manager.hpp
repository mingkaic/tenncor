
#include <future>

#include <boost/bimap.hpp>
#include <boost/lexical_cast.hpp>

#include "distrib/manager.hpp"
#include "dbg/distr_ext/print/service.hpp"

#ifndef DISTRIB_DBG_MANAGER_HPP
#define DISTRIB_DBG_MANAGER_HPP

namespace distr
{

struct DistrDbgManager final : public iDistrManager
{
	DistrDbgManager (
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
		printsvc_ = std::make_unique<DistrPrintService>(consul_, cfg, builder, iosvc_.get());

		server_ = builder.BuildAndStart();
		teq::infof("[server %s] listening on %s", consul_.id_.c_str(), address.c_str());

		iosvc_->initialize_server_call(*cq_);
		opsvc_->initialize_server_call(*cq_);
		printsvc_->initialize_server_call(*cq_);

		for (size_t i = 0; i < nthreads; ++i)
		{
			rpc_jobs_.push_back(std::thread(
				&DistrDbgManager::handle_rpcs, this));
		}
	}

	~DistrDbgManager (void)
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

	DistrPrintService& get_print (void)
	{
		return *printsvc_;
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

	std::unique_ptr<DistrPrintService> printsvc_;

	std::unique_ptr<grpc::ServerCompletionQueue> cq_;

	std::unique_ptr<grpc::Server> server_;

	std::vector<std::thread> rpc_jobs_;
};

using DistDbgMgrptrT = std::shared_ptr<DistrDbgManager>;

}

#endif // DISTRIB_DBG_MANAGER_HPP
