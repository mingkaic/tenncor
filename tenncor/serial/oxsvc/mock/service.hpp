
#ifndef DISTR_OXSVC_MOCK_SERVICE_HPP
#define DISTR_OXSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/serial/oxsvc/oxsvc.hpp"

struct MockOxService final : public distr::ox::iSerializeService
{
	grpc::Service* get_service (void) override
	{
		return nullptr;
	}

	void RequestGetSaveGraph (grpc::ServerContext* ctx,
		distr::ox::GetSaveGraphRequest* req,
		egrpc::iResponder<distr::ox::GetSaveGraphResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::ox::GetSaveGraphResponse>*>(&writer);
		assert(nullptr != mock_res);
		save_packets_.push_back(ServicePacket<
			distr::ox::GetSaveGraphRequest,
			MockResponder<distr::ox::GetSaveGraphResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	void RequestPostLoadGraph (grpc::ServerContext* ctx,
		distr::ox::PostLoadGraphRequest* req,
		egrpc::iResponder<distr::ox::PostLoadGraphResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::ox::PostLoadGraphResponse>*>(&writer);
		assert(nullptr != mock_res);
		load_packets_.push_back(ServicePacket<
			distr::ox::PostLoadGraphRequest,
			MockResponder<distr::ox::PostLoadGraphResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
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
		auto svc = MockServerBuilder::get_service<MockOxService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ox service found in %s", address_.c_str());
		}
		auto packet = svc->save_packets_.front();
		svc->save_packets_.pop_front();
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

	grpc::Status PostLoadGraph (grpc::ClientContext* context,
		const distr::ox::PostLoadGraphRequest& request,
		distr::ox::PostLoadGraphResponse* response) override
	{
		auto svc = MockServerBuilder::get_service<MockOxService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock ox service found in %s", address_.c_str());
		}
		auto packet = svc->load_packets_.front();
		svc->load_packets_.pop_front();
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
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::ox::GetSaveGraphResponse>(
		[this, context, &request](distr::ox::GetSaveGraphResponse* response)
		{
			return this->GetSaveGraph(context, request, response);
		}, *mcq);
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
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::ox::PostLoadGraphResponse>(
		[this, context, &request](distr::ox::PostLoadGraphResponse* response)
		{
			return this->PostLoadGraph(context, request, response);
		}, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_oxsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OXSVC_MOCK_SERVICE_HPP
