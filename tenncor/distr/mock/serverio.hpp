
#ifndef DISTR_MOCK_SERVERIO_HPP
#define DISTR_MOCK_SERVERIO_HPP

#include "tenncor/distr/distr.hpp"

#include <google/protobuf/util/json_util.h>

struct MockCQueue final : public egrpc::iCQueue
{
	static std::unordered_map<grpc::CompletionQueue*,MockCQueue*> real2mock_;

	MockCQueue (void) :
		cq_aggregation_(
		[this]()
		{
			void* tag;
			bool ok = true;
			while (cq_.Next(&tag, &ok))
			{
				this->write(tag);
			}
			shutdown_ = true;
			cond_.notify_all();
		})
	{
		real2mock_.emplace(&cq_, this);
	}

	~MockCQueue (void)
	{
		shutdown();
		cq_aggregation_.join();
		real2mock_.erase(&cq_);
	}

	bool next (void** ptr, bool* ok) override
	{
		std::unique_lock<std::mutex> lk(mtx_);
		cond_.wait(lk,
		[this]()
		{
			return q_.size() > 0 || shutdown_;
		});
		bool exit = q_.size() > 0;
		if (q_.size() > 0)
		{
			auto result = q_.front();
			q_.pop_front();
			*ptr = result.first;
			*ok = result.second;
		}
		else
		{
			*ok = false;
		}
		lk.unlock();
		return exit && !shutdown_;
	}

	void shutdown (void) override
	{
		cq_.Shutdown();
	}

	grpc::CompletionQueue* get_cq (void) override
	{
		return &cq_;
	}

	void write (void* ptr, bool success = true)
	{
		{
			std::lock_guard<std::mutex> lk(mtx_);
			q_.push_back({ptr, success});
		}
		cond_.notify_all();
	}

	grpc::CompletionQueue cq_;

	std::condition_variable cond_;

	std::mutex mtx_;

	std::list<std::pair<void*,bool>> q_;

	std::atomic<bool> shutdown_ = false;

	std::thread cq_aggregation_;
};

struct MockServer final : public distr::iServer
{
	MockServer (std::string addr, egrpc::iCQueue& q,
		std::unordered_set<distr::iService*> services) :
		addr_(addr), q_(&q), services_(services) {}

	void shutdown (void) override {}

	std::string addr_;

	egrpc::iCQueue* q_;

	std::unordered_set<distr::iService*> services_;
};

struct MockServerBuilder final : public distr::iServerBuilder
{
	static types::StrUMapT<MockServer*> mock_servers_;

	static std::mutex mock_mtx_;

	template <typename SVC>
	static SVC* get_service (const std::string& address)
	{
		std::lock_guard<std::mutex> lk(mock_mtx_);
		auto server = estd::must_getf(mock_servers_, address,
			"cannot find server %s", address.c_str());
		for (auto service : server->services_)
		{
			if (auto svc = dynamic_cast<SVC*>(service))
			{
				return svc;
			}
		}
		return nullptr;
	}

	static void add_service (const std::string& address, MockServer* server)
	{
		std::lock_guard<std::mutex> lk(mock_mtx_);
		mock_servers_.erase(address);
		mock_servers_.emplace(address, server);
	}

	static void clear_service (void)
	{
		std::lock_guard<std::mutex> lk(mock_mtx_);
		mock_servers_.clear();
	}

	distr::iServerBuilder& register_service (distr::iService& service) override
	{
		services_.emplace(&service);
		return *this;
	}

	distr::iServerBuilder& add_listening_port (
		const std::string& address,
		std::shared_ptr<grpc::ServerCredentials> creds,
		int* selected_port = nullptr) override
	{
		last_address_ = address;
		return *this;
	}

	distr::CQueueptrT add_completion_queue (
		bool is_frequently_polled = true) override
	{
		return std::make_unique<MockCQueue>();
	}

	std::unique_ptr<distr::iServer> build_and_start (void) override
	{
		auto server = new MockServer(last_address_, *last_q_, services_);
		last_server_ = server;
		MockServerBuilder::add_service(last_address_, server);
		return std::unique_ptr<distr::iServer>(server);
	}

	std::string last_address_;

	egrpc::iCQueue* last_q_ = nullptr;

	std::unordered_set<distr::iService*> services_;

	MockServer* last_server_ = nullptr;
};

template <typename REQ, typename RES>
struct ServicePacket
{
	REQ* req_ = nullptr;

	RES* res_ = nullptr;

	egrpc::iServerCall* call_ = nullptr;
};

template <typename R>
struct MockResponder final : public egrpc::iResponder<R>
{
	void finish (const R& res, grpc::Status status, void* tag) override
	{
		reply_.MergeFrom(res);
		status_ = status_;
		tag_ = tag;
	}

	void finish_with_error (grpc::Status status, void* tag) override
	{
		status_ = status_;
		tag_ = tag;
	}

	R reply_;

	grpc::Status status_;

	void* tag_;
};

template <typename R>
struct MockWriter final : public egrpc::iWriter<R>
{
	void write (const R& res, void* tag) override
	{
		replies_.push_back(res);
	}

	void finish (grpc::Status status, void* tag) override
	{
		status_ = status_;
		tag_ = tag;
	}

	std::list<R> replies_;

	grpc::Status status_;

	void* tag_;
};

template <typename R>
struct MockClientAsyncResponseReader final : public grpc::ClientAsyncResponseReaderInterface<R>
{
	using CallerF = std::function<grpc::Status(R*)>;

	MockClientAsyncResponseReader (CallerF caller, egrpc::iCQueue& cq) :
		caller_(caller), cq_(&static_cast<MockCQueue&>(cq)) {}

	void StartCall (void) override {}

	void ReadInitialMetadata (void* tag) override {}

	void Finish (R* msg, grpc::Status* status, void* tag) override
	{
		std::thread([=, this]
		{
			*status = caller_(msg);
			cq_->write(tag);
		}).detach();
	}

	CallerF caller_;

	MockCQueue* cq_;
};

template <typename REQ, typename RES>
struct MockClientAsyncReader final : public grpc::ClientAsyncReaderInterface<RES>
{
	MockClientAsyncReader (ServicePacket<REQ,MockWriter<RES>>& packet,
		egrpc::iCQueue& cq) : packet_(packet),
		cq_(&static_cast<MockCQueue&>(cq)) {}

	void StartCall (void* tag) override
	{
		cq_->write(tag);
	}

	void ReadInitialMetadata (void* tag) override {}

	void Read (RES* msg, void* tag) override
	{
		packet_.call_->serve();
		if (packet_.res_->replies_.empty())
		{
			cq_->write(tag, false);
			return;
		}
		auto& res = packet_.res_->replies_.front();
		msg->MergeFrom(res);
		packet_.res_->replies_.pop_front();
		cq_->write(tag);
	}

	void Finish (grpc::Status* status, void* tag) override
	{
		packet_.call_->serve();
		*status = packet_.res_->status_;
		cq_->write(tag);
	}

	ServicePacket<REQ,MockWriter<RES>> packet_;

	MockCQueue* cq_;
};

#endif // DISTR_MOCK_SERVERIO_HPP
