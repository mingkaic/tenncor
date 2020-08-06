
#include "error/error.hpp"

#include "distrib/consul.hpp"
#include "distrib/egrpc/client_async.hpp"
#include "distrib/egrpc/iclient.hpp"

#ifndef DISTRIB_PEER_SVC_HPP
#define DISTRIB_PEER_SVC_HPP

namespace distr
{

template <typename CLI> // CLI has base egrpc::GrpcClient
struct PeerService
{
	PeerService (ConsulService& consul,
		const egrpc::ClientConfig& cfg,
		size_t nthreads = 3) :
		consul_(consul), cfg_(cfg)
	{
		update_clients();
		for (size_t i = 0; i < nthreads; ++i)
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
		if (consul_.id_ == peer_id)
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
		auto peers = consul_.get_peers();
		for (auto peer : peers)
		{
			if (false == estd::has(clients_, peer.first))
			{
				clients_.insert({peer.first, std::make_unique<CLI>(
					grpc::CreateChannel(peer.second,
						grpc::InsecureChannelCredentials()), cfg_,
					consul_.id_ + "->" + peer.first)});
			}
		}
	}

	egrpc::ClientConfig cfg_;

	ConsulService& consul_;

	grpc::CompletionQueue cq_;

	estd::StrMapT<std::unique_ptr<CLI>> clients_; // todo: add cleanup job for clients

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

}

#endif // DISTRIB_PEER_SVC_HPP
