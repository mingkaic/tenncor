
#ifndef DISTR_IPEER_SVC_HPP
#define DISTR_IPEER_SVC_HPP

#include "egrpc/egrpc.hpp"

namespace distr
{

#define SVC_RES_DECL(FNAME, REQ, RES)\
virtual void FNAME (grpc::ServerContext* ctx, REQ* req,\
	egrpc::iResponder<RES>& writer,\
	egrpc::iCQueue& cq, void* tag) = 0;

#define SVC_STREAM_DECL(FNAME, REQ, RES)\
virtual void FNAME (grpc::ServerContext* ctx, REQ* req,\
	egrpc::iWriter<RES>& writer,\
	egrpc::iCQueue& cq, void* tag) = 0;

#define SVC_RES_DEFN(FNAME, REQ, RES)\
void FNAME (grpc::ServerContext* ctx, REQ* req,\
	egrpc::iResponder<RES>& writer,\
	egrpc::iCQueue& cq, void* tag) override { auto scq = estd::must_cast<\
	grpc::ServerCompletionQueue>(cq.get_cq()); svc_.FNAME(ctx, req, \
	&static_cast<egrpc::GrpcResponder<RES>&>(writer).responder_, scq, scq, tag); }

#define SVC_STREAM_DEFN(FNAME, REQ, RES)\
void FNAME (grpc::ServerContext* ctx, REQ* req,\
	egrpc::iWriter<RES>& writer,\
	egrpc::iCQueue& cq, void* tag) override { auto scq = estd::must_cast<\
	grpc::ServerCompletionQueue>(cq.get_cq()); svc_.FNAME(ctx, req, \
	&static_cast<egrpc::GrpcWriter<RES>&>(writer).writer_, scq, scq, tag); }

struct iService
{
	virtual ~iService (void) = default;

	virtual grpc::Service* get_service (void) = 0;
};

struct iServer
{
	virtual ~iServer (void) = default;

	virtual void shutdown (void) = 0;
};

using CQueueptrT = std::unique_ptr<egrpc::iCQueue>;

struct iServerBuilder
{
	virtual ~iServerBuilder (void) = default;

	virtual iServerBuilder& register_service (iService& service) = 0;

	virtual iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) = 0;

	virtual CQueueptrT add_completion_queue (
		bool is_frequently_polled = true) = 0;

	virtual std::unique_ptr<iServer> build_and_start (void) = 0;
};

struct iPeerService
{
	virtual ~iPeerService (void) = default;

	virtual void register_service (iServerBuilder& builder) = 0;

	virtual void initialize_server_call (egrpc::iCQueue& cq) = 0;
};

}

#endif // DISTR_IPEER_SVC_HPP
