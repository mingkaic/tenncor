
#ifndef DISTR_MOCK_P2P_HPP
#define DISTR_MOCK_P2P_HPP

#include "tenncor/distr/p2p.hpp"

struct MockP2P : public distr::iP2PService
{
	MockP2P (std::mutex& kv_mtx,
		const std::string& local_id,
		const std::string& address,
		types::StrUMapT<std::string>& peers,
		types::StrUMapT<std::string>& shared_kv) :
		kv_mtx_(&kv_mtx), id_(local_id), address_(address),
		kv_(&shared_kv), address_book_(&peers) {}

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
};

#endif // DISTR_MOCK_P2P_HPP
