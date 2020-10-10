
#ifndef DISTR_MOCK_P2P_HPP
#define DISTR_MOCK_P2P_HPP

#include "tenncor/distr/p2p.hpp"

#include "tenncor/distr/mock/distr.health.grpc.pb.h"

struct HealthyP2PService : public distr::iP2PService
{
	HealthyP2PService (void)
	{
	}

	virtual ~HealthyP2PService (void) = default;

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		global::infof("[server %s] health id: %s", get_local_peer().c_str(), health_id_.c_str());
		// CheckHealth
		using CheckHealthCallT = egrpc::AsyncServerCall<distr::health::CheckHealthRequest,distr::health::CheckHealthResponse>;
		auto chealth_logger = std::make_shared<global::FormatLogger>(
			global::get_logger(), fmts::sprintf("[server %s:CheckHealth] ",
			get_local_peer().c_str()));
		new CheckHealthCallT(chealth_logger,
		[this](grpc::ServerContext* ctx, distr::health::CheckHealthRequest* req,
			grpc::ServerAsyncResponseWriter<distr::health::CheckHealthResponse>* writer,
			grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
			void* tag)
		{
			this->service_.RequestCheckHealth(ctx, req, writer, cq, ccq, tag);
		},
		[this](const distr::health::CheckHealthRequest& req, distr::health::CheckHealthResponse& res)
		{
			res.set_uuid(health_id_);
			return grpc::Status::OK;
		}, &cq);
	}

	std::string get_health_id (void) const
	{
		return health_id_;
	}

private:
	distr::health::DistrHealth::AsyncService service_;

	std::string health_id_ = fmts::sprintf("%p", this);
};

struct MockP2P : public HealthyP2PService
{
	MockP2P (std::mutex& kv_mtx,
		const std::string& local_id,
		const std::string& address,
		types::StrUMapT<std::string>& peers,
		types::StrUMapT<std::string>& shared_kv,
		types::StrUMapT<std::string>& health_ids) :
		kv_mtx_(&kv_mtx), id_(local_id), address_(address),
		kv_(&shared_kv), address_book_(&peers),
		health_ids_(&health_ids) {}

	types::StrUMapT<std::string> get_peers (void) override
	{
		auto out = *address_book_;
		out.erase(get_local_peer());
		return out;
	}

	void set_kv (
		const std::string& key, const std::string& value) override
	{
		std::lock_guard<std::mutex> guard(*kv_mtx_);
		kv_->emplace(key, value);
	}

	std::string get_kv (
		const std::string& key, const std::string& default_val) override
	{
		std::lock_guard<std::mutex> guard(*kv_mtx_);
		return estd::try_get(*kv_, key, default_val);
	}

	std::string get_local_peer (void) const override
	{
		return id_;
	}

	std::string get_local_addr (void) const override
	{
		return address_;
	}

	std::mutex* kv_mtx_;

	std::string id_;

	std::string address_;

	types::StrUMapT<std::string>* kv_;

	types::StrUMapT<std::string>* address_book_;

	types::StrUMapT<std::string>* health_ids_;
};

error::ErrptrT check_health (const std::string& address,
	const std::string& health_id, size_t nretry = 1);

#endif // DISTR_MOCK_P2P_HPP
