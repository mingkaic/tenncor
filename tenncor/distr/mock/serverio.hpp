
#ifndef DISTR_MOCK_SERVERIO_HPP
#define DISTR_MOCK_SERVERIO_HPP

#include "tenncor/distr/distr.hpp"

struct MockServer final : public distr::iServer
{
	MockServer (std::string addr, grpc::ServerCompletionQueue* q,
		types::StrUMapT<distr::iService*> services) :
		addr_(addr), q_(q), services_(services) {}

	void shutdown (void) override {}

	std::string addr_;

	grpc::ServerCompletionQueue* q_;

	types::StrUMapT<distr::iService*> services_;
};

struct MockServerBuilder final : public distr::iServerBuilder
{
	distr::iServerBuilder& register_service (distr::iService& service) override
	{
		services_.emplace("", &service);
		return *this;
	}

	distr::iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) override
	{
		last_address_ = address;
		return *this;
	}

	std::unique_ptr<grpc::ServerCompletionQueue>
	add_completion_queue (bool is_frequently_polled = true) override
	{
		auto q = mock_builder_.AddCompletionQueue(is_frequently_polled);
		last_q_ = q.get();
		return std::move(q);
	}

	std::unique_ptr<distr::iServer> build_and_start (void) override
	{
		auto server = new MockServer(last_address_, last_q_, services_);
		last_server_ = server;
		return std::unique_ptr<distr::iServer>(server);
	}

	grpc::ServerBuilder mock_builder_;

	std::string last_address_;

	grpc::ServerCompletionQueue* last_q_ = nullptr;

	types::StrUMapT<distr::iService*> services_;

	MockServer* last_server_ = nullptr;
};

#endif // DISTR_MOCK_SERVERIO_HPP
