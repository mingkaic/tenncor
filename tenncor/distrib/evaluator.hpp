#include <boost/bimap.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/lexical_cast.hpp>

#include "distrib/client.hpp"
#include "distrib/server.hpp"
#include "distrib/consul.hpp"

#ifndef DISTRIB_EVALUATOR_HPP
#define DISTRIB_EVALUATOR_HPP

namespace distr
{

const std::string default_service = "tenncor";

const std::string node_lookup_prefix = "tenncor.node.";

using UuidSetT = std::unordered_set<std::string>;

struct DistrEvaluator final : public iDistrEvaluator
{
	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	DistrEvaluator (PartDerF derive,
		ppconsul::Consul& consul, size_t port,
		const std::string& service = default_service,
		const std::string& id = "",
		const ClientConfig& cfg = ClientConfig()) :
		consul_(consul, port, (id.empty() ?
			boost::uuids::to_string(DistrEvaluator::uuid_gen_()) : id), service),
		server_(this, derive, port, consul_.id_),
		cli_cfg_(cfg)
	{
		update_clients();

		std::thread rpc_job(&DistrEvaluator::handle_rpcs, this);
		rpc_job.detach();
	}

	~DistrEvaluator (void)
	{
		data_queue_.Shutdown();
	}

	/// Implementation of iDistrEvaluator
	void expose_node (teq::TensptrT tens) override
	{
		if (estd::has(owners_, tens.get()) &&
			false == owners_[tens.get()].expired())
		{
			return;
		}
		std::string id = boost::uuids::to_string(DistrEvaluator::uuid_gen_());
		shared_nodes_.insert({id, tens.get()});
		owners_.emplace(tens.get(), tens);
		consul_.set_kv(node_lookup_prefix + id, consul_.id_);
	}

	/// Implementation of iDistrEvaluator
	std::optional<std::string> lookup_id (teq::TensptrT tens) const override
	{
		std::optional<std::string> out;
		if (estd::has(shared_nodes_.right, tens.get()))
		{
			out = shared_nodes_.right.at(tens.get());
		}
		return out;
	}

	/// Implementation of iDistrEvaluator
	teq::TensptrT lookup_node (error::ErrptrT& err,
		const std::string& id, bool recursive = true) override
	{
		if (estd::has(shared_nodes_.left, id))
		{
			auto existing = shared_nodes_.left.at(id);
			if (false == owners_[existing].expired())
			{
				return owners_[existing].lock();
			}
			err = error::errorf("node id %s is expired", id.c_str());
			return nullptr;
		}
		if (false == recursive)
		{
			err = error::errorf(
				"no id %s found: will not recurse", id.c_str());
			return nullptr;
		}
		std::string peer_id = consul_.get_kv(node_lookup_prefix + id, "");
		if (peer_id.empty())
		{
			err = error::errorf("no peer found for node %s",
				(node_lookup_prefix + id).c_str());
			return nullptr;
		}
		if (false == estd::has(clients_, peer_id))
		{
			update_clients(); // get specific peer
		}
		FindNodesRequest req;
		FindNodesResponse res;
		req.add_uuids(id);
		auto status = clients_[peer_id]->lookup_node(req, res);
		if (false == status.ok())
		{
			err = error::errorf("grpc status not ok: %s ()",
				status.error_message().c_str());
			return nullptr;
		}
		if (res.values().empty())
		{
			err = error::errorf("no result found in received peer '%s'",
				peer_id.c_str());
			return nullptr;
		}
		auto node = res.values().at(0);
		auto node_id = node.uuid();
		auto& slist = node.shape();
		std::vector<teq::DimT> sdims(slist.begin(), slist.end());
		auto ref = std::make_shared<DistRef>(
			egen::get_type(node.dtype()),
			teq::Shape(sdims),
			node.instance(), node_id);
		shared_nodes_.insert({node_id, ref.get()});
		owners_.emplace(ref.get(), ref);
		remotes_.emplace(ref);
		return ref;
	}

	/// Implementation of iDistrEvaluator
	std::string get_id (void) const override
	{
		return consul_.id_;
	}

	/// Implementation of iDistrEvaluator
	DRefptrSetT get_remotes (void) const override
	{
		return remotes_;
	}

private:
	/// Implementation of iDistrEvaluator
	std::future<void> call (const std::string& cluster_id,
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

		GetDataRequest req;
		google::protobuf::RepeatedPtrField<std::string>
		uuids(node_ids.begin(), node_ids.end());
		req.mutable_uuids()->Swap(&uuids);
		return clients_.at(cluster_id)->get_data(
			data_queue_, req, shared_nodes_);
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

	boost::bimap<std::string,teq::iTensor*> shared_nodes_;

	estd::StrMapT<DistrCliPtrT> clients_; // todo: clean-up clients

	DRefptrSetT remotes_;

	teq::OwnerMapT owners_;

	ConsulService consul_;

	DistrServer server_;

	ClientConfig cli_cfg_;
};

using DEvalptrT = std::shared_ptr<DistrEvaluator>;

}

#endif // DISTRIB_EVALUATOR_HPP
