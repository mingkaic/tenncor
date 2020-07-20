#include <boost/bimap.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/lexical_cast.hpp>

#include "distrib/client.hpp"
#include "distrib/server.hpp"
#include "distrib/consul.hpp"

#ifndef DISTRIB_SESSION_HPP
#define DISTRIB_SESSION_HPP

namespace distrib
{

const std::string default_service = "tenncor";

const std::string node_lookup_prefix = "tenncor.node.";

using UuidSetT = std::unordered_set<std::string>;

struct DistribSess final : public iDistribSess
{
	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	DistribSess (ppconsul::Consul& consul, size_t port,
		const std::string& service = default_service,
		const std::string& id = "",
		const ClientConfig& cfg = ClientConfig()) :
		consul_(consul, port, (id.empty() ?
			boost::uuids::to_string(DistribSess::uuid_gen_()) : id), service),
		server_(this, port, consul_.id_),
		cli_cfg_(cfg)
	{
		update_clients();

		std::thread rpc_job(&DistribSess::handle_rpcs, this);
		rpc_job.detach();
	}

	~DistribSess (void)
	{
		data_queue_.Shutdown();
	}

	/// Implementation of iDistribSess
	std::optional<std::string> lookup_id (teq::TensptrT tens) const override
	{
		std::optional<std::string> out;
		if (estd::has(shared_nodes_.right, tens))
		{
			out = shared_nodes_.right.at(tens);
		}
		return out;
	}

	/// Implementation of iDistribSess
	teq::TensptrT lookup_node (err::ErrptrT& err,
		const std::string& id, bool recursive = true) override
	{
		if (false == estd::has(shared_nodes_.left, id))
		{
			if (false == recursive)
			{
				err = std::make_shared<err::ErrMsg>(
					"no id %s found: will not recurse", id.c_str());
				return nullptr;
			}
			std::string peer_id = consul_.get_kv(node_lookup_prefix + id, "");
			if (peer_id.empty())
			{
				err = std::make_shared<err::ErrMsg>(
					"no peer found for node %s",
					(node_lookup_prefix + id).c_str());
				return nullptr;
			}
			if (false == estd::has(clients_, peer_id))
			{
				update_clients(); // get specific peer
			}
			distr::FindNodesRequest req;
			distr::FindNodesResponse res;
			req.add_uuids(id);
			auto status = clients_[peer_id]->lookup_node(req, res);
			if (false == status.ok())
			{
				err = std::make_shared<err::ErrMsg>(
					"grpc status not ok: %s ()",
					status.error_message().c_str());
				return nullptr;
			}
			if (res.values().empty())
			{
				err = std::make_shared<err::ErrMsg>(
					"no result found in received peer '%s'", peer_id.c_str());
				return nullptr;
			}
			auto node = res.values().at(0);
			auto node_id = node.uuid();
			auto& slist = node.shape();
			std::vector<teq::DimT> sdims(slist.begin(), slist.end());
			shared_nodes_.insert({node_id,
				std::make_shared<DistRef>(
					egen::get_type(node.dtype()),
					teq::Shape(sdims),
					node.instance(), node_id)});
		}
		return shared_nodes_.left.at(id);
	}

	/// Implementation of iDistribSess
	std::string get_id (void) const override
	{
		return consul_.id_;
	}

protected:
	/// Implementation of iDistribSess
	std::future<void> call (
		const std::string& cluster_id,
		const std::unordered_set<std::string>& node_ids) override
	{
		if (false == estd::has(clients_, cluster_id))
		{
			teq::fatalf("cannot find client %s", cluster_id.c_str());
		}
		if (std::any_of(node_ids.begin(), node_ids.end(),
			[this](const std::string& id)
			{
				return false == estd::has(this->shared_nodes_.left, id);
			}))
		{
			teq::fatalf("some nodes from %s not found locally",
				cluster_id.c_str());
		}

		distr::GetDataRequest req;
		google::protobuf::RepeatedPtrField<std::string>
		uuids(node_ids.begin(), node_ids.end());
		req.mutable_uuids()->Swap(&uuids);
		return clients_.at(cluster_id)->get_data(
			data_queue_, req, shared_nodes_);
	}

private:
	void store_tracked (const teq::TensptrSetT& locals) override
	{
		for (teq::TensptrT local : locals)
		{
			std::string id = boost::uuids::to_string(DistribSess::uuid_gen_());
			shared_nodes_.insert({id, local});
			consul_.set_kv(node_lookup_prefix + id, consul_.id_);
		}
	}

	void update_clients (void)
	{
		auto peers = consul_.get_peers();
		for (auto peer : peers)
		{
			if (false == estd::has(clients_, peer.first))
			{
				clients_.insert({peer.first, std::make_unique<DistrCli>(
					grpc::CreateChannel(peer.second,
						grpc::InsecureChannelCredentials()),
					consul_.id_ + "->" + peer.first, cli_cfg_)});
			}
		}
	}

	void handle_rpcs (void)
	{
		void* got_tag;
		bool ok = true;
		while (data_queue_.Next(&got_tag, &ok))
		{
			auto handler = static_cast<iCliRespHandler*>(got_tag);
			handler->handle(ok);
		}
	}

	grpc::CompletionQueue data_queue_;

	boost::bimap<std::string,teq::TensptrT> shared_nodes_;

	std::unordered_map<std::string,DistrCliPtrT> clients_; // todo: clean-up clients

	ConsulService consul_;

	AsyncDistrServer server_;
	// DistrServer server_;

	ClientConfig cli_cfg_;
};

}

#endif // DISTRIB_SESSION_HPP
