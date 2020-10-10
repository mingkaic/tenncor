
#ifndef DISTR_PEER_SVC_HPP
#define DISTR_PEER_SVC_HPP

#include "error/error.hpp"

#include "tenncor/distr/p2p.hpp"

namespace distr
{

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

template <typename CLI> // CLI has base egrpc::GrpcClient
struct PeerService : public iPeerService
{
	PeerService (const PeerServiceConfig& cfg) :
		cli_(cfg.cli_), p2p_(cfg.p2p_)
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
				clients_.insert({peer.first, std::make_unique<CLI>(
					grpc::CreateChannel(peer.second,
						grpc::InsecureChannelCredentials()), cli_,
					get_peer_id() + "->" + peer.first)});
			}
		}
	}

	std::string get_peer_id (void) const
	{
		return p2p_->get_local_peer();
	}

	egrpc::ClientConfig cli_;

	iP2PService* p2p_;

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

using RegisterSvcF = std::function<error::ErrptrT(\
	estd::ConfigMap<>&,const PeerServiceConfig&)>;

}

#endif // DISTR_PEER_SVC_HPP
