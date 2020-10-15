
#ifndef DISTR_IPEER_SVC_HPP
#define DISTR_IPEER_SVC_HPP

#include "egrpc/egrpc.hpp"

namespace distr
{

#define SVC_RES_DECL(FNAME, REQ, RES)\
virtual void FNAME (grpc::ServerContext* ctx, REQ* req,\
	grpc::ServerAsyncResponseWriter<RES>* writer,\
	grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,\
	void* tag) = 0;

#define SVC_STREAM_DECL(FNAME, REQ, RES)\
virtual void FNAME (grpc::ServerContext* ctx, REQ* req,\
	grpc::ServerAsyncWriter<RES>* writer,\
	grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,\
	void* tag) = 0;

#define SVC_RES_DEFN(FNAME, REQ, RES)\
void FNAME (grpc::ServerContext* ctx, REQ* req,\
	grpc::ServerAsyncResponseWriter<RES>* writer,\
	grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,\
	void* tag) override { svc_.FNAME(ctx, req, writer, cq, ccq, tag); }

#define SVC_STREAM_DEFN(FNAME, REQ, RES)\
void FNAME (grpc::ServerContext* ctx, REQ* req,\
	grpc::ServerAsyncWriter<RES>* writer,\
	grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,\
	void* tag) override { svc_.FNAME(ctx, req, writer, cq, ccq, tag); }

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

struct iServerBuilder
{
	virtual ~iServerBuilder (void) = default;

	virtual iServerBuilder& register_service (iService& service) = 0;

	virtual iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) = 0;

	virtual std::unique_ptr<grpc::ServerCompletionQueue>
	add_completion_queue (bool is_frequently_polled = true) = 0;

	virtual std::unique_ptr<iServer> build_and_start (void) = 0;
};

struct iPeerService
{
	virtual ~iPeerService (void) = default;

	virtual void register_service (iServerBuilder& builder) = 0;

	virtual void initialize_server_call (grpc::ServerCompletionQueue& cq) = 0;
};

}

#endif // DISTR_IPEER_SVC_HPP
