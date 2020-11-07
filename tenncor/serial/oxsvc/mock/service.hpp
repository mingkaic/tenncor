
#ifndef DISTR_OXSVC_MOCK_SERVICE_HPP
#define DISTR_OXSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"

struct MockOxService final : public distr::ox::iSerializeService
{
	~MockOxService (void)
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

	void RequestGetSaveGraph (grpc::ServerContext* ctx,
		distr::ox::GetSaveGraphRequest* req,
		egrpc::iResponder<distr::ox::GetSaveGraphResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::ox::GetSaveGraphResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		save_packets_.push_back(ServicePacket<
			distr::ox::GetSaveGraphRequest,
			MockResponder<distr::ox::GetSaveGraphResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	void RequestPostLoadGraph (grpc::ServerContext* ctx,
		distr::ox::PostLoadGraphRequest* req,
		egrpc::iResponder<distr::ox::PostLoadGraphResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::ox::PostLoadGraphResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		load_packets_.push_back(ServicePacket<
			distr::ox::PostLoadGraphRequest,
			MockResponder<distr::ox::PostLoadGraphResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	egrpc::RespondptrT<distr::ox::GetSaveGraphResponse>
	make_get_save_graph_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::ox::GetSaveGraphResponse>>();
	}

	egrpc::RespondptrT<distr::ox::PostLoadGraphResponse>
	make_post_load_graph_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::ox::PostLoadGraphResponse>>();
	}

	ServicePacket<distr::ox::GetSaveGraphRequest,
		MockResponder<distr::ox::GetSaveGraphResponse>>
	save_depacket (void)
	{
		auto out = save_packets_.front();
		save_packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

	ServicePacket<distr::ox::PostLoadGraphRequest,
		MockResponder<distr::ox::PostLoadGraphResponse>>
	load_depacket (void)
	{
		auto out = load_packets_.front();
		load_packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

	std::list<ServicePacket<distr::ox::GetSaveGraphRequest,
		MockResponder<distr::ox::GetSaveGraphResponse>>> save_packets_;

	std::list<ServicePacket<distr::ox::PostLoadGraphRequest,
		MockResponder<distr::ox::PostLoadGraphResponse>>> load_packets_;
};

struct MockOxStub final : public distr::ox::DistrSerialization::StubInterface
{
	MockOxStub (const std::string& address) : address_(address) {}

	grpc::Status GetSaveGraph (grpc::ClientContext* context,
		const distr::ox::GetSaveGraphRequest& request,
		distr::ox::GetSaveGraphResponse* response) override
	{
		return grpc::Status::OK;
	}

	grpc::Status PostLoadGraph (grpc::ClientContext* context,
		const distr::ox::PostLoadGraphRequest& request,
		distr::ox::PostLoadGraphResponse* response) override
	{
		return grpc::Status::OK;
	}

private:
	grpc::ClientAsyncResponseReaderInterface<distr::ox::GetSaveGraphResponse>*
	AsyncGetSaveGraphRaw (grpc::ClientContext* context,
		const distr::ox::GetSaveGraphRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncGetSaveGraphRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::ox::GetSaveGraphResponse>*
	PrepareAsyncGetSaveGraphRaw (grpc::ClientContext* context,
		const ::distr::ox::GetSaveGraphRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOxService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ox service found in %s", address_.c_str());
		}
		auto packet = svc->save_depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::ox::GetSaveGraphResponse>(
			packet.res_, packet.call_, *mcq);
	}

	grpc::ClientAsyncResponseReaderInterface<distr::ox::PostLoadGraphResponse>*
	AsyncPostLoadGraphRaw (grpc::ClientContext* context,
		const distr::ox::PostLoadGraphRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncPostLoadGraphRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::ox::PostLoadGraphResponse>*
	PrepareAsyncPostLoadGraphRaw (grpc::ClientContext* context,
		const ::distr::ox::PostLoadGraphRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOxService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ox service found in %s", address_.c_str());
		}
		auto packet = svc->load_depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::ox::PostLoadGraphResponse>(
			packet.res_, packet.call_, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_oxsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OXSVC_MOCK_SERVICE_HPP
