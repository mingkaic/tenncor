
#ifndef DISTR_IO_DATA_HPP
#define DISTR_IO_DATA_HPP

#include <shared_mutex>

#include <boost/bimap.hpp>
#include <boost/lexical_cast.hpp>

#include "tenncor/distr/reference.hpp"
#include "tenncor/distr/p2p.hpp"

namespace distr
{

namespace io
{

const std::string node_lookup_prefix = "tenncor.node.";

const size_t nretry_getkv = 3;

using OptIDT = std::optional<std::string>;

struct DistrIOData
{
	DistrIOData (iP2PService* consul) :
		consul_(consul) {}

	std::string cache_tens (teq::TensptrT tens,
		const OptIDT& suggested_id = OptIDT())
	{
		auto tensptr = tens.get();
		{
			std::shared_lock<std::shared_mutex> read_guard(smtx_);
			if (auto ref = std::dynamic_pointer_cast<iDistrRef>(tens))
			{
				if (false == estd::has(owners_, ref.get()))
				{
					owners_.emplace(ref.get(), ref);
				}
				if (false == estd::has(shareds_.left, ref->node_id()))
				{
					shareds_.insert({ref->node_id(), ref.get()});
					remotes_.emplace(ref);
				}
				return ref->node_id();
			}
			if (false == estd::has(owners_, tensptr))
			{
				owners_.emplace(tensptr, tens);
			}
			if (estd::has(shareds_.right, tensptr))
			{
				return shareds_.right.at(tensptr);
			}
		}
		std::string id;
		if (suggested_id && false == bool(get_peer(*suggested_id)))
		{
			id = *suggested_id;
		}
		else
		{
			if (suggested_id)
			{
				global::warnf("suggested id %s already exists, will "
					"use auto-generating id", suggested_id->c_str());
			}
			id = global::get_generator()->get_str();
		}
		consul_->set_kv(node_lookup_prefix + id, consul_->get_local_peer());
		std::lock_guard<std::shared_mutex> write_guard(smtx_);
		shareds_.insert({id, tensptr});
		return id;
	}

	OptIDT get_id (teq::iTensor* tens) const
	{
		OptIDT out;
		std::shared_lock<std::shared_mutex> read_guard(smtx_);
		if (estd::has(shareds_.right, tens))
		{
			out = shareds_.right.at(tens);
		}
		return out;
	}

	teq::TensptrT get_tens (const std::string& id) const
	{
		teq::TensptrT out = nullptr;
		std::shared_lock<std::shared_mutex> read_guard(smtx_);
		if (estd::has(shareds_.left, id))
		{
			auto existing = shareds_.left.at(id);
			out = owners_.at(existing);
		}
		return out;
	}

	OptIDT get_peer (const std::string& id) const
	{
		OptIDT out;
		std::string key = node_lookup_prefix + id;
		std::string peer_id = "";
		for (size_t i = 0; i < nretry_getkv && peer_id.empty(); ++i)
		{
			peer_id = consul_->get_kv(key, "");
		}
		if (peer_id.size() > 0)
		{
			out = peer_id;
		}
		return out;
	}

	DRefptrSetT get_remotes (void) const
	{
		return remotes_;
 	}

private:
	mutable std::shared_mutex smtx_;

	iP2PService* consul_;

	DRefptrSetT remotes_;

	teq::OwnMapT owners_;

	boost::bimap<std::string,teq::iTensor*> shareds_;
};

}

}

#endif // DISTR_IO_DATA_HPP
