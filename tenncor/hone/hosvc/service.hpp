
#ifndef DISTRIB_HO_SERVICE_HPP
#define DISTRIB_HO_SERVICE_HPP

#include "tenncor/distr/imanager.hpp"

#include "tenncor/hone/hone.hpp"
#include "tenncor/hone/hosvc/client.hpp"

namespace distr
{

namespace ho
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

using HoServiceT = DistrOptimization::AsyncService;

const std::string hosvc_key = "distr_hosvc";

struct DistrHoService final : public PeerService<DistrHoCli>
{
	DistrHoService (const PeerServiceConfig& cfg) :
		PeerService<DistrHoCli>(cfg), iosvc_(iosvc) {}

	void optimize (const opt::Optimization& optimize)
	{
		//
	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// PutOptimize
		auto popt_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:PutOptimize] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerCall<PutOptimizeRequest,
			PutOptimizeResponse>(lnodes_logger,
			[this](grpc::ServerContext* ctx, PutOptimizeRequest* req,
				grpc::ServerAsyncResponseWriter<PutOptimizeResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestPutOptimize(ctx, req, writer, cq, ccq, tag);
			},
			[this](const PutOptimizeRequest& req, PutOptimizeResponse& res)
			{
				auto alias = fmts::sprintf("%s:PutOptimize", get_peer_id().c_str());
				return grpc::Status::OK;
			}, &cq);
	}

private:
	DistrOptimization::AsyncService service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_hosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

ho::DistrHoService& get_hosvc (iDistrManager& manager);

}

#endif // DISTRIB_HO_SERVICE_HPP
