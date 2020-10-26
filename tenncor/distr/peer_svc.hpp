
#ifndef DISTR_PEER_SVC_HPP
#define DISTR_PEER_SVC_HPP

#include "error/error.hpp"

#include "tenncor/distr/p2p.hpp"
#include "tenncor/distr/ipeer_svc.hpp"

namespace distr
{

struct GrpcServer final : public iServer
{
	GrpcServer (std::unique_ptr<grpc::Server>&& server) :
		server_(std::move(server)) {}

	void shutdown (void) override
	{
		server_->Shutdown();
	}

	std::unique_ptr<grpc::Server> server_;
};

struct ServerBuilder final : public iServerBuilder
{
	iServerBuilder& register_service (iService& service) override
	{
		builder_.RegisterService(service.get_service());
		return *this;
	}

	iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) override
	{
		builder_.AddListeningPort(address, creds, selected_port);
		return *this;
	}

	CQueueptrT add_completion_queue (bool is_frequently_polled = true) override
	{
		return std::make_unique<egrpc::GrpcServerCQueue>(
			builder_.AddCompletionQueue(is_frequently_polled));
	}

	std::unique_ptr<iServer> build_and_start (void) override
	{
		return std::make_unique<GrpcServer>(builder_.BuildAndStart());
	}

	grpc::ServerBuilder builder_;
};

struct PeerServiceConfig
{
	PeerServiceConfig (iP2PService* p2p,
		const egrpc::ClientConfig& cli,
		size_t nthreads = 3) :
		nthreads_(nthreads), p2p_(p2p), cli_(cli) {}

	size_t nthreads_;

	iP2PService* p2p_;

	egrpc::ClientConfig cli_;
};

struct iClientBuilder
{
	virtual ~iClientBuilder (void) = default;

	virtual egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config, const std::string& alias) const = 0;

	virtual CQueueptrT build_cqueue (void) const = 0;
};

using CliBuildptrT = std::shared_ptr<iClientBuilder>;

template <typename CLI>
struct ClientBuilder final : public iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config, const std::string& alias) const override
	{
		return new CLI(grpc::CreateChannel(addr,
			grpc::InsecureChannelCredentials()),
			config, alias);
	}

	CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<egrpc::GrpcCQueue>();
	}
};

template <typename CLI> // CLI has base egrpc::GrpcClient
struct PeerService : public iPeerService
{
	PeerService (const PeerServiceConfig& cfg,
		CliBuildptrT builder = std::make_shared<ClientBuilder<CLI>>()) :
		cli_(cfg.cli_), p2p_(cfg.p2p_),
		builder_(builder), cq_(builder->build_cqueue())
	{
		update_clients();
		// if nthread_ == 0, use 1 thread anyways
		for (size_t i = 0, nlimits = cfg.nthreads_ > 0 ? cfg.nthreads_ : 1;
			i < nlimits; ++i)
		{
			cli_jobs_.push_back(std::move(std::thread(
				&PeerService::handle_clients, this)));
		}
	}

	virtual ~PeerService (void)
	{
		cq_->shutdown();
		for (auto& cli_job : cli_jobs_)
		{
			cli_job.join();
		}
	}

protected:
	CLI* get_client (
		error::ErrptrT& err,
		const std::string& peer_id)
	{
		err = nullptr;
		if (get_peer_id() == peer_id)
		{
			err = error::errorf("cannot get client for local server %s",
				peer_id.c_str());
			return nullptr;
		}
		// try to find clients before giving up and declaring can't find
		if (false == estd::has(clients_, peer_id))
		{
			update_clients(); // get specific peer
		}
		if (false == estd::has(clients_, peer_id))
		{
			err = error::errorf("cannot find client %s", peer_id.c_str());
			return nullptr;
		}
		return clients_.at(peer_id).get();
	}

	void update_clients (void)
	{
		auto peers = p2p_->get_peers();
		for (auto peer : peers)
		{
			if (false == estd::has(clients_, peer.first))
			{
				clients_.insert({
					peer.first, std::unique_ptr<CLI>(
						static_cast<CLI*>(builder_->build_client(peer.second,
							cli_, get_peer_id() + "->" + peer.first)))
				});
			}
		}
	}

	std::string get_peer_id (void) const
	{
		return p2p_->get_local_peer();
	}

	egrpc::ClientConfig cli_;

	iP2PService* p2p_;

	CliBuildptrT builder_;

	CQueueptrT cq_;

	types::StrUMapT<std::unique_ptr<CLI>> clients_; // todo: add cleanup job for clients

private:
	void handle_clients (void)
	{
		void* got_tag = nullptr;
		bool ok = true;
		while (cq_->next(&got_tag, &ok))
		{
			if (nullptr != got_tag)
			{
				auto handler = static_cast<egrpc::iClientHandler*>(got_tag);
				handler->handle(ok);
				got_tag = nullptr;
			}
		}
	}

	std::vector<std::thread> cli_jobs_;
};

using RegisterSvcF = std::function<error::ErrptrT(\
	estd::ConfigMap<>&,const PeerServiceConfig&)>;

}

#endif // DISTR_PEER_SVC_HPP
