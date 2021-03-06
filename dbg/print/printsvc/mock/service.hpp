
#ifndef DISTR_PRINTSVC_MOCK_SERVICE_HPP
#define DISTR_PRINTSVC_MOCK_SERVICE_HPP

#include "tenncor/distr/mock/mock.hpp"

#include "dbg/print/printsvc/printsvc.hpp"

struct MockPrintService final : public distr::print::iPrintService
{
	~MockPrintService (void)
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

	void RequestListAscii (grpc::ServerContext* ctx,
		distr::print::ListAsciiRequest* req,
		egrpc::iWriter<distr::print::AsciiEntry>& writer,
		egrpc::iCQueue& cq, void* tag) override
	{
		auto call = static_cast<egrpc::iServerCall*>(tag);
		auto mock_res = dynamic_cast<MockWriter<
			distr::print::AsciiEntry>*>(&writer);
		assert(nullptr != mock_res);
		mock_res->set_cq(static_cast<MockSrvCQT&>(cq));
		packets_.push_back(ServicePacket<
			distr::print::ListAsciiRequest,
			MockWriter<distr::print::AsciiEntry>>{
			req, mock_res, call
		});
		calls_.emplace(call);
	}

	egrpc::WriterptrT<distr::print::AsciiEntry>
	make_list_ascii_writer (grpc::ServerContext& ctx) const override
	{
		return std::make_unique<MockWriter<distr::print::AsciiEntry>>();
	}

	ServicePacket<distr::print::ListAsciiRequest,
		MockWriter<distr::print::AsciiEntry>>
	depacket (void)
	{
		auto out = packets_.front();
		packets_.pop_front();
		calls_.erase(out.call_);
		return out;
	}

private:
	std::unordered_set<egrpc::iServerCall*> calls_;

	std::list<ServicePacket<distr::print::ListAsciiRequest,
		MockWriter<distr::print::AsciiEntry>>> packets_;
};

struct MockPrintStub final : public distr::print::DistrPrint::StubInterface
{
	MockPrintStub (const std::string& address) : address_(address) {}

private:
	grpc::ClientReaderInterface<distr::print::AsciiEntry>*
	ListAsciiRaw (grpc::ClientContext* context,
		const distr::print::ListAsciiRequest& request) override
	{
		return nullptr;
	}

	grpc::ClientAsyncReaderInterface<distr::print::AsciiEntry>*
	AsyncListAsciiRaw (grpc::ClientContext* context,
		const distr::print::ListAsciiRequest& request,
		grpc::CompletionQueue* cq, void* tag) override
	{
		auto out = PrepareAsyncListAsciiRaw(context, request, cq);
		out->StartCall(tag);
		return out;
	}

	grpc::ClientAsyncReaderInterface<distr::print::AsciiEntry>*
	PrepareAsyncListAsciiRaw (grpc::ClientContext* context,
		const distr::print::ListAsciiRequest& request,
		grpc::CompletionQueue* cq) override
	{
		auto mcq = estd::must_getf(MockCliCQT::real_to_mock(), cq,
			"cannot find grpc completion queue %p", cq);
		auto svc = MockServerBuilder::get_service<MockPrintService>(address_);
		if (nullptr == svc)
		{
			global::fatalf("no mock print service found in %s", address_.c_str());
		}
		auto packet = svc->depacket();
		packet.req_->MergeFrom(request);
		return new MockClientAsyncReader<distr::print::AsciiEntry>(
			packet.res_, packet.call_, *mcq);
	}

	std::string address_;
};

error::ErrptrT register_mock_printsvc (estd::ConfigMap<>& svcs,
	const distr::PeerServiceConfig& cfg);

#endif // DISTR_PRINTSVC_MOCK_SERVICE_HPP
