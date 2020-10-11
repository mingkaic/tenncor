
#ifndef DISTR_MOCK_SERVICE_HPP
#define DISTR_MOCK_SERVICE_HPP

#include "tenncor/distr/distr.hpp"

struct MockClient
{
	MockClient (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) {}
};

struct MockService : public distr::PeerService<MockClient>
{
	MockService (const distr::PeerServiceConfig& cfg) :
		PeerService<MockClient>(cfg) {}

	void register_service (grpc::ServerBuilder& builder) override
	{
		++registry_count_;
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		++initial_count_;
	}

	MockClient* public_client (error::ErrptrT& err, const std::string& peer_id)
	{
		return get_client(err, peer_id);
	}

	size_t registry_count_ = 0;

	size_t initial_count_ = 0;
};

#endif // DISTR_MOCK_SERVICE_HPP
