
#ifndef DISTR_PRINT_SERVICE_HPP
#define DISTR_PRINT_SERVICE_HPP

#include "egrpc/egrpc.hpp"

#include "tenncor/distr/iosvc/service.hpp"

#include "dbg/print/printsvc/data.hpp"
#include "dbg/print/printsvc/client.hpp"

namespace distr
{

namespace print
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

const std::string printsvc_key = "dbg_printsvc";

struct iPrintService : public iService
{
	virtual ~iPrintService (void) = default;

	SVC_STREAM_DECL(RequestListAscii, ListAsciiRequest, AsciiEntry)
};

struct PrintService final : public iPrintService
{
	grpc::Service* get_service (void) override
	{
		return &svc_;
	}

	SVC_STREAM_DEFN(RequestListAscii, ListAsciiRequest, AsciiEntry)

	DistrPrint::AsyncService svc_;
};

struct DistrPrintService final : public PeerService<DistrPrintCli>
{
	DistrPrintService (const PeerServiceConfig& cfg,
		io::DistrIOService* iosvc,
		const PrintEqConfig& printopts = PrintEqConfig(),
		PeerService<DistrPrintCli>::BuildCliF builder =
			PeerService<DistrPrintCli>::default_builder,
		std::shared_ptr<iPrintService> svc =
			std::make_shared<PrintService>()) :
		PeerService<DistrPrintCli>(cfg, builder),
		iosvc_(iosvc), printopts_(printopts), service_(svc)
	{
		assert(nullptr != service_);
	}

	void print_ascii (std::ostream& os, teq::iTensor* tens);

	void register_service (iServerBuilder& builder) override
	{
		builder.register_service(*service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// ListAscii
		using ListAsciiCallT = egrpc::AsyncServerStreamCall<ListAsciiRequest,AsciiEntry,types::StringsT>;
		auto lascii_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:ListAscii] ",
				get_peer_id().c_str()));
		new ListAsciiCallT(lascii_logger,
		[this](grpc::ServerContext* ctx, ListAsciiRequest* req,
			grpc::ServerAsyncWriter<AsciiEntry>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_->RequestListAscii(ctx, req, writer, cq, ccq, tag);
		},
		[this](types::StringsT& states, const ListAsciiRequest& req)
		{
			return this->startup_list_ascii(states, req);
		},
		[this](const ListAsciiRequest& req,
			types::StringsT::iterator& it, AsciiEntry& reply)
		{
			return this->process_list_ascii(req, it, reply);
		}, &cq);
	}

private:
	grpc::Status startup_list_ascii (
		types::StringsT& roots,
		const ListAsciiRequest& req)
	{
		auto& uuids = req.uuids();
		roots.insert(roots.end(), uuids.begin(), uuids.end());
		return grpc::Status::OK;
	}

	bool process_list_ascii (const ListAsciiRequest& req,
		types::StringsT::iterator& it, AsciiEntry& reply);

	AsciiRemotesT print_ascii_remotes (const AsciiRemotesT& remotes);

	DistrPrintData cache_;

	io::DistrIOService* iosvc_;

	PrintEqConfig printopts_;

	size_t depthlimit_ = 10;

	std::shared_ptr<iPrintService> service_;
};

#undef _ERR_CHECK

}

error::ErrptrT register_printsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

print::DistrPrintService& get_printsvc (iDistrManager& manager);

}

#endif // DISTR_PRINT_SERVICE_HPP
