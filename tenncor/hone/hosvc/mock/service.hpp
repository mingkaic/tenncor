
#ifndef DISTR_HOSVC_MOCK_SERVICE_HPP
#define DISTR_HOSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/hone/hosvc/hosvc.hpp"

#include <google/protobuf/util/json_util.h>

struct MockHoService final : public distr::ho::iHoService
{
	grpc::Service* get_service (void) override
	{
		return nullptr;
	}

	void RequestPutOptimize (grpc::ServerContext* ctx,
		distr::ho::PutOptimizeRequest* req,
		egrpc::iResponder<distr::ho::PutOptimizeResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::ho::PutOptimizeResponse>*>(&writer);
		assert(nullptr != mock_res);
		packets_.push_back(ServicePacket<
			distr::ho::PutOptimizeRequest,
			MockResponder<distr::ho::PutOptimizeResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	egrpc::RespondptrT<distr::ho::PutOptimizeResponse>
	make_put_optimize_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::ho::PutOptimizeResponse>>();
	}

	std::list<ServicePacket<distr::ho::PutOptimizeRequest,
		MockResponder<distr::ho::PutOptimizeResponse>>> packets_;
};

struct MockHoStub final : public distr::ho::DistrOptimization::StubInterface
{
	MockHoStub (const std::string& address) : address_(address) {}

	grpc::Status PutOptimize (grpc::ClientContext* context,
		const distr::ho::PutOptimizeRequest& request,
		distr::ho::PutOptimizeResponse* response) override
	{
		auto svc = MockServerBuilder::get_service<MockHoService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ho service found in %s", address_.c_str());
		}
		auto packet = svc->packets_.front();
		svc->packets_.pop_front();
		packet.req_->MergeFrom(request);
		packet.call_->serve();
		auto responder = packet.res_;
		if (!responder->status_.ok())
		{
			return responder->status_;
		}
		response->MergeFrom(responder->reply_);
		return grpc::Status::OK;
	}

private:
	grpc::ClientAsyncResponseReaderInterface<distr::ho::PutOptimizeResponse>*
	AsyncPutOptimizeRaw (grpc::ClientContext* context,
		const distr::ho::PutOptimizeRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncPutOptimizeRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::ho::PutOptimizeResponse>*
	PrepareAsyncPutOptimizeRaw (grpc::ClientContext* context,
		const ::distr::ho::PutOptimizeRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::ho::PutOptimizeResponse>(
		[this, context, &request](distr::ho::PutOptimizeResponse* response)
		{
			return this->PutOptimize(context, request, response);
		}, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_hosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OPSVC_MOCK_SERVICE_HPP
