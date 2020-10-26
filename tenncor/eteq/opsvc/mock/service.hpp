
#ifndef DISTR_OPSVC_MOCK_SERVICE_HPP
#define DISTR_OPSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/eteq/opsvc/opsvc.hpp"

struct MockOpService final : public distr::op::iOpService
{
	grpc::Service* get_service (void) override
	{
		return nullptr;
	}

	void RequestGetData (grpc::ServerContext* ctx,
		distr::op::GetDataRequest* req,
		egrpc::iWriter<distr::op::NodeData>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockWriter<
			distr::op::NodeData>*>(&writer);
		assert(nullptr != mock_res);
		data_packets_.push_back(ServicePacket<
			distr::op::GetDataRequest,
			MockWriter<distr::op::NodeData>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	void RequestListReachable (grpc::ServerContext* ctx,
		distr::op::ListReachableRequest* req,
		egrpc::iResponder<distr::op::ListReachableResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::op::ListReachableResponse>*>(&writer);
		assert(nullptr != mock_res);
		reachable_packets_.push_back(ServicePacket<
			distr::op::ListReachableRequest,
			MockResponder<distr::op::ListReachableResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	void RequestCreateDerive (grpc::ServerContext* ctx,
		distr::op::CreateDeriveRequest* req,
		egrpc::iResponder<distr::op::CreateDeriveResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto mock_res = dynamic_cast<MockResponder<
			distr::op::CreateDeriveResponse>*>(&writer);
		assert(nullptr != mock_res);
		derive_packets_.push_back(ServicePacket<
			distr::op::CreateDeriveRequest,
			MockResponder<distr::op::CreateDeriveResponse>>{
			req, mock_res,
			static_cast<egrpc::iServerCall*>(tag)
		});
	}

	egrpc::WriterptrT<distr::op::NodeData>
	make_get_data_writer (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockWriter<distr::op::NodeData>>();
	}

	egrpc::RespondptrT<distr::op::ListReachableResponse>
	make_list_reachable_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::op::ListReachableResponse>>();
	}

	egrpc::RespondptrT<distr::op::CreateDeriveResponse>
	make_create_derive_responder (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockResponder<distr::op::CreateDeriveResponse>>();
	}

	std::list<ServicePacket<distr::op::GetDataRequest,
		MockWriter<distr::op::NodeData>>> data_packets_;

	std::list<ServicePacket<distr::op::ListReachableRequest,
		MockResponder<distr::op::ListReachableResponse>>> reachable_packets_;

	std::list<ServicePacket<distr::op::CreateDeriveRequest,
		MockResponder<distr::op::CreateDeriveResponse>>> derive_packets_;
};

struct MockOpStub final : public distr::op::DistrOperation::StubInterface
{
	MockOpStub (const std::string& address) : address_(address) {}

	grpc::Status ListReachable (grpc::ClientContext* context,
		const distr::op::ListReachableRequest& request,
		distr::op::ListReachableResponse* response) override
	{
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->reachable_packets_.front();
		svc->reachable_packets_.pop_front();
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

	grpc::Status CreateDerive (grpc::ClientContext* context,
		const distr::op::CreateDeriveRequest& request,
		distr::op::CreateDeriveResponse* response) override
	{
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->derive_packets_.front();
		svc->derive_packets_.pop_front();
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
	grpc::ClientReaderInterface<distr::op::NodeData>*
	GetDataRaw (grpc::ClientContext* context,
		const distr::op::GetDataRequest& request) override
	{
		return nullptr;
	}

	grpc::ClientAsyncReaderInterface<distr::op::NodeData>*
	AsyncGetDataRaw (grpc::ClientContext* context,
		const distr::op::GetDataRequest& request,
		grpc::CompletionQueue* cq, void* tag) override
	{
		auto out = PrepareAsyncGetDataRaw(context, request, cq);
		out->StartCall(tag);
		return out;
	}

	grpc::ClientAsyncReaderInterface<distr::op::NodeData>*
	PrepareAsyncGetDataRaw (grpc::ClientContext* context,
		const distr::op::GetDataRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->data_packets_.front();
		svc->data_packets_.pop_front();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncReader<
			distr::op::GetDataRequest,
			distr::op::NodeData>(packet, *mcq);
	}

	grpc::ClientAsyncResponseReaderInterface<distr::op::ListReachableResponse>*
	AsyncListReachableRaw (grpc::ClientContext* context,
		const distr::op::ListReachableRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncListReachableRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::op::ListReachableResponse>*
	PrepareAsyncListReachableRaw (grpc::ClientContext* context,
		const ::distr::op::ListReachableRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::op::ListReachableResponse>(
		[this, context, &request](distr::op::ListReachableResponse* response)
		{
			return this->ListReachable(context, request, response);
		}, *mcq);
	}

	grpc::ClientAsyncResponseReaderInterface<distr::op::CreateDeriveResponse>*
	AsyncCreateDeriveRaw (grpc::ClientContext* context,
		const distr::op::CreateDeriveRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto out = PrepareAsyncCreateDeriveRaw(context, request, cq);
		out->StartCall();
		return out;
	}

	grpc::ClientAsyncResponseReaderInterface<distr::op::CreateDeriveResponse>*
	PrepareAsyncCreateDeriveRaw (grpc::ClientContext* context,
		const ::distr::op::CreateDeriveRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCQueue::real2mock_, cq,
			"cannot find grpc completion queue %p", cq);
		return new MockClientAsyncResponseReader<distr::op::CreateDeriveResponse>(
		[this, context, &request](distr::op::CreateDeriveResponse* response)
		{
			return this->CreateDerive(context, request, response);
		}, *mcq);
	}

	std::string address_;
};

struct MockDistrOpCliBuilder final : public distr::iClientBuilder
{
	egrpc::GrpcClient* build_client (const std::string& addr,
		const egrpc::ClientConfig& config,
		const std::string& alias) const override
	{
		return new distr::op::DistrOpCli(new MockOpStub(addr), config, alias);
	}

	distr::CQueueptrT build_cqueue (void) const override
	{
		return std::make_unique<MockCQueue>();
	}
};

error::ErrptrT register_mock_opsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OPSVC_MOCK_SERVICE_HPP
