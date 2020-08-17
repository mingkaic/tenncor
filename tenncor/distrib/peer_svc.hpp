
#include "error/error.hpp"

#include "distrib/consul.hpp"
#include "egrpc/client_async.hpp"
#include "egrpc/iclient.hpp"

#ifndef DISTRIB_PEER_SVC_HPP
#define DISTRIB_PEER_SVC_HPP

namespace distr
{

struct iPeerService
{
	virtual ~iPeerService (void) = default;

	virtual void register_service (grpc::ServerBuilder& builder) = 0;

	virtual void initialize_server_call (grpc::ServerCompletionQueue& cq) = 0;
};

struct PeerServiceConfig
{
	PeerServiceConfig (ConsulService* consul,
		const egrpc::ClientConfig& cli,
		size_t nthreads = 3) :
		nthreads_(nthreads), consul_(consul), cli_(cli) {}

	size_t nthreads_;

	ConsulService* consul_;

	egrpc::ClientConfig cli_;
};

template <typename CLI> // CLI has base egrpc::GrpcClient
struct PeerService : public iPeerService
{
	PeerService (const PeerServiceConfig& cfg) :
		consul_(cfg.consul_), cli_(cfg.cli_)
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
		cq_.Shutdown();
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
		auto peers = consul_->get_peers();
		for (auto peer : peers)
		{
			if (false == estd::has(clients_, peer.first))
			{
				clients_.insert({peer.first, std::make_unique<CLI>(
					grpc::CreateChannel(peer.second,
						grpc::InsecureChannelCredentials()), cli_,
					get_peer_id() + "->" + peer.first)});
			}
		}
	}

	std::string get_peer_id (void) const
	{
		return consul_->id_;
	}

	egrpc::ClientConfig cli_;

	ConsulService* consul_;

	grpc::CompletionQueue cq_;

	types::StrUMapT<std::unique_ptr<CLI>> clients_; // todo: add cleanup job for clients

private:
	void handle_clients (void)
	{
		void* got_tag;
		bool ok = true;
		while (cq_.Next(&got_tag, &ok))
		{
			auto handler = static_cast<egrpc::iClientHandler*>(got_tag);
			handler->handle(ok);
		}
	}

	std::vector<std::thread> cli_jobs_;
};

template <typename T>
void wait_on_future (std::future<T>& done)
{
	while (done.valid() && done.wait_for(
		std::chrono::milliseconds(1)) ==
		std::future_status::timeout);
}

}

#endif // DISTRIB_PEER_SVC_HPP
