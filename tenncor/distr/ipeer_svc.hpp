
#ifndef DISTR_IPEER_SVC_HPP
#define DISTR_IPEER_SVC_HPP

#include "egrpc/egrpc.hpp"

namespace distr
{

struct iServerBuilder
{
	virtual ~iServerBuilder (void) = default;

	virtual iServerBuilder& register_service (grpc::Service* service) = 0;

	virtual iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) = 0;

	virtual std::unique_ptr<grpc::ServerCompletionQueue>
	add_completion_queue (bool is_frequently_polled = true) = 0;

	virtual std::unique_ptr<grpc::Server> build_and_start (void) = 0;
};

struct iPeerService
{
	virtual ~iPeerService (void) = default;

	virtual void register_service (iServerBuilder& builder) = 0;

	virtual void initialize_server_call (grpc::ServerCompletionQueue& cq) = 0;
};

}

#endif // DISTR_IPEER_SVC_HPP
