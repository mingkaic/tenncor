
#ifndef DISTR_OPSVC_MOCK_SERVICE_HPP
#define DISTR_OPSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/eteq/opsvc/opsvc.hpp"

struct MockOpService final : public distr::op::iOpService
{
	~MockOpService (void)
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

	void RequestGetData (grpc::ServerContext* ctx,
		distr::op::GetDataRequest* req,
		egrpc::iWriter<distr::op::NodeData>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockWriter<
			distr::op::NodeData>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		data_packets_.push_back(ServicePacket<
			distr::op::GetDataRequest,
			MockWriter<distr::op::NodeData>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	void RequestListReachable (grpc::ServerContext* ctx,
		distr::op::ListReachableRequest* req,
		egrpc::iResponder<distr::op::ListReachableResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::op::ListReachableResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		reachable_packets_.push_back(ServicePacket<
			distr::op::ListReachableRequest,
			MockResponder<distr::op::ListReachableResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	void RequestCreateDerive (grpc::ServerContext* ctx,
		distr::op::CreateDeriveRequest* req,
		egrpc::iResponder<distr::op::CreateDeriveResponse>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockResponder<
			distr::op::CreateDeriveResponse>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		derive_packets_.push_back(ServicePacket<
			distr::op::CreateDeriveRequest,
			MockResponder<distr::op::CreateDeriveResponse>>{
			req, mock_res, call
		});
		calls_.emplace(call);
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

	ServicePacket<distr::op::GetDataRequest,
		MockWriter<distr::op::NodeData>>
	data_depacket (void)
	{
		auto out = data_packets_.front();
		data_packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

	ServicePacket<distr::op::ListReachableRequest,
		MockResponder<distr::op::ListReachableResponse>>
	reachable_depacket (void)
	{
		auto out = reachable_packets_.front();
		reachable_packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

	ServicePacket<distr::op::CreateDeriveRequest,
		MockResponder<distr::op::CreateDeriveResponse>>
	derive_depacket (void)
	{
		auto out = derive_packets_.front();
		derive_packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

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
		return grpc::Status::OK;
	}

	grpc::Status CreateDerive (grpc::ClientContext* context,
		const distr::op::CreateDeriveRequest& request,
		distr::op::CreateDeriveResponse* response) override
	{
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
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->data_depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncReader<distr::op::NodeData>(
			packet.res_, packet.call_, *mcq);
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
		const distr::op::ListReachableRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->reachable_depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::op::ListReachableResponse>(
			packet.res_, packet.call_, *mcq);
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
		const distr::op::CreateDeriveRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockOpService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock op service found in %s", address_.c_str());
		}
		auto packet = svc->derive_depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncResponseReader<distr::op::CreateDeriveResponse>(
			packet.res_, packet.call_, *mcq);
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
		return std::make_unique<MockCliCQT>();
	}
};

error::ErrptrT register_mock_opsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_OPSVC_MOCK_SERVICE_HPP
