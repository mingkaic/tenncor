
#include "eigen/random.hpp"

#include "distrib/reference.hpp"
#include "distrib/consul.hpp"

#ifndef DISTRIB_IO_DATA_HPP
#define DISTRIB_IO_DATA_HPP

namespace distr
{

const std::string node_lookup_prefix = "tenncor.node.";

struct DistrIOData
{
	DistrIOData (ConsulService& consul) :
		consul_(consul) {}

	std::string cache_tens (teq::TensptrT tens)
	{
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
		auto tensptr = tens.get();
		if (false == estd::has(owners_, tensptr))
		{
			owners_.emplace(tensptr, tens);
		}
		if (estd::has(shareds_.right, tensptr))
		{
			return shareds_.right.at(tensptr);
		}
		std::string id = boost::uuids::to_string(eigen::rand_uuid_gen()());
		shareds_.insert({id, tensptr});
		consul_.set_kv(node_lookup_prefix + id, consul_.id_);
		return id;
	}

	std::optional<std::string> get_id (teq::iTensor* tens) const
	{
		std::optional<std::string> out;
		if (estd::has(shareds_.right, tens))
		{
			out = shareds_.right.at(tens);
		}
		return out;
	}

	teq::TensptrT get_tens (const std::string& id) const
	{
		teq::TensptrT out = nullptr;
		if (estd::has(shareds_.left, id))
		{
			auto existing = shareds_.left.at(id);
			out = owners_.at(existing);
		}
		return out;
	}

	std::optional<std::string> get_peer (const std::string& id) const
	{
		std::optional<std::string> out;
		std::string peer_id = consul_.get_kv(node_lookup_prefix + id, "");
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
	ConsulService& consul_;

	DRefptrSetT remotes_;

	teq::OwnMapT owners_;

	boost::bimap<std::string,teq::iTensor*> shareds_;
};

}

#endif // DISTRIB_IO_DATA_HPP
