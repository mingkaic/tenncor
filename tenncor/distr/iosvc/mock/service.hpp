
#ifndef DISTR_IOSVC_MOCK_SERVICE_HPP
#define DISTR_IOSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"

struct MockIOService final : public distr::io::iIOService
{
	~MockIOService (void)
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

	void RequestListNodes (grpc::ServerContext* ctx, distr::io::ListNodesRequest* req,
		egrpc::iResponder<distr::io::ListNodesResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::io::ListNodesResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		packets_.push_back(ServicePacket<
			distr::io::ListNodesRequest,
			MockResponder<distr::io::ListNodesResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	egrpc::RespondptrT<distr::io::ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::io::ListNodesResponse>>();
	}

	ServicePacket<distr::io::ListNodesRequest,
		MockResponder<distr::io::ListNodesResponse>>
	depacket (void)
	{
		auto out = packets_.front();
		packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

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
		const distr::io::ListNodesRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockIOService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock io service found in %s", address_.c_str());
		}
		auto packet = svc->depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::io::ListNodesResponse>(
			packet.res_, packet.call_, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_iosvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_IOSVC_MOCK_SERVICE_HPP
