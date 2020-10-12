
#ifndef DISTR_IPEER_SVC_HPP
#define DISTR_IPEER_SVC_HPP

#include "egrpc/egrpc.hpp"

namespace distr
{

struct iPeerService
{
	virtual ~iPeerService (void) = default;

	virtual void register_service (grpc::ServerBuilder& builder) = 0;

	virtual void initialize_server_call (grpc::ServerCompletionQueue& cq) = 0;
};

}

#endif // DISTR_IPEER_SVC_HPP
