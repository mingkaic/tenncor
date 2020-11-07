
#ifndef DISTR_LUSVC_MOCK_SERVICE_HPP
#define DISTR_LUSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/find/lusvc/lusvc.hpp"

struct MockLuService final : public distr::lu::iLuService
{
	~MockLuService (void)
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

	void RequestListNodes (grpc::ServerContext* ctx, distr::lu::ListNodesRequest* req,
		egrpc::iResponder<distr::lu::ListNodesResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::lu::ListNodesResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		packets_.push_back(ServicePacket<
			distr::lu::ListNodesRequest,
			MockResponder<distr::lu::ListNodesResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	egrpc::RespondptrT<distr::lu::ListNodesResponse>
	make_list_nodes_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::lu::ListNodesResponse>>();
	}

	ServicePacket<distr::lu::ListNodesRequest,
		MockResponder<distr::lu::ListNodesResponse>>
	depacket (void)
	{
		auto out = packets_.front();
		packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

	std::list<ServicePacket<distr::lu::ListNodesRequest,
		MockResponder<distr::lu::ListNodesResponse>>> packets_;
};

struct MockLuStub final : public distr::lu::DistrLookup::StubInterface
{
	MockLuStub (const std::string& address) : address_(address) {}

	grpc::Status ListNodes (grpc::ClientContext* context,
		const distr::lu::ListNodesRequest& request,
		distr::lu::ListNodesResponse* response) override
	{
		return grpc::Status::OK;
	}

private:
	grpc::ClientAsyncResponseReaderInterface<distr::lu::ListNodesResponse>*
	AsyncListNodesRaw (grpc::ClientContext* context,
		const distr::lu::ListNodesRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncListNodesRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::lu::ListNodesResponse>*
	PrepareAsyncListNodesRaw (grpc::ClientContext* context,
		const distr::lu::ListNodesRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completlun queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockLuService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock lu service found in %s", address_.c_str());
		}
		auto packet = svc->depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::lu::ListNodesResponse>(
			packet.res_, packet.call_, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_lusvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_LUSVC_MOCK_SERVICE_HPP
