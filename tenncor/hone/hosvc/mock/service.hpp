
#ifndef DISTR_HOSVC_MOCK_SERVICE_HPP
#define DISTR_HOSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/hone/hosvc/hosvc.hpp"

#include <google/protobuf/util/json_util.h>

struct MockHoService final : public distr::ho::iHoService
{
	~MockHoService (void)
	{
		for (auto call : calls_)
		{
			delete call;
		}
	}

	grpc::Service* get_service (void) override
	{
		return nullptr;
	}

	void RequestPutOptimize (grpc::ServerContext* ctx,
		distr::ho::PutOptimizeRequest* req,
		egrpc::iResponder<distr::ho::PutOptimizeResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::ho::PutOptimizeResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		packets_.push_back(ServicePacket<
			distr::ho::PutOptimizeRequest,
			MockResponder<distr::ho::PutOptimizeResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	egrpc::RespondptrT<distr::ho::PutOptimizeResponse>
	make_put_optimize_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::ho::PutOptimizeResponse>>();
	}

	ServicePacket<distr::ho::PutOptimizeRequest,
		MockResponder<distr::ho::PutOptimizeResponse>>
	depacket (void)
	{
		auto out = packets_.front();
		packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

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
		const distr::ho::PutOptimizeRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockHoService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ho service found in %s", address_.c_str());
		}
		auto packet = svc->depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::ho::PutOptimizeResponse>(
			packet.res_, packet.call_, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_hosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OPSVC_MOCK_SERVICE_HPP
