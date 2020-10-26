
#ifndef DISTR_IOSVC_MOCK_SERVICE_HPP
#define DISTR_IOSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"

struct MockIOService final : public distr::io::iIOService
{
	grpc::Service* get_service (void) override
	{
		return nullptr;
	}

	void RequestListNodes (grpc::ServerContext* ctx, distr::io::ListNodesRequest* req,
		egrpc::iResponder<distr::io::ListNodesResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::io::ListNodesResponse>*>(&writer);
		assert(nullptr != mock_res);
		packets_.push_back(ServicePacket<
			distr::io::ListNodesRequest,
			MockResponder<distr::io::ListNodesResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	egrpc::RespondptrT<distr::io::ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::io::ListNodesResponse>>();
	}

	std::list<ServicePacket<distr::io::ListNodesRequest,
		MockResponder<distr::io::ListNodesResponse>>> packets_;
};

struct MockIOStub final : public distr::io::DistrInOut::StubInterface
{
	MockIOStub (const std::string& address) : address_(address) {}

	grpc::Status ListNodes (grpc::ClientContext* context,
		const distr::io::ListNodesRequest& request,
		distr::io::ListNodesResponse* response) override
	{
		auto svc = MockServerBuilder::get_service<MockIOService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock io service found in %s", address_.c_str());
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
	grpc::ClientAsyncResponseReaderInterface<distr::io::ListNodesResponse>*
	AsyncListNodesRaw (grpc::ClientContext* context,
		const distr::io::ListNodesRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncListNodesRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::io::ListNodesResponse>*
	PrepareAsyncListNodesRaw (grpc::ClientContext* context,
		const ::distr::io::ListNodesRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::io::ListNodesResponse>(
		[this, context, &request](distr::io::ListNodesResponse* response)
		{
			return this->ListNodes(context, request, response);
		}, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_iosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_IOSVC_MOCK_SERVICE_HPP
