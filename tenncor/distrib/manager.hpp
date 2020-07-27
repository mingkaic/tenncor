
#include <boost/bimap.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/lexical_cast.hpp>

#include "distrib/client.hpp"
#include "distrib/server.hpp"
#include "distrib/consul.hpp"

#ifndef DISTRIB_MANAGER_HPP
#define DISTRIB_MANAGER_HPP

namespace distr
{

const std::string default_service = "tenncor";

const std::string node_lookup_prefix = "tenncor.node.";

struct DistManager : public iDistManager
{
	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	DistManager (
		ppconsul::Consul& consul, size_t port,
		const std::string& service = default_service,
		const std::string& id = "",
		const ClientConfig& cfg = ClientConfig()) :
		consul_(consul, port, (id.empty() ?
			boost::uuids::to_string(DistManager::uuid_gen_()) : id), service),
		server_(this, port, consul_.id_),
		cli_cfg_(cfg)
	{
		update_clients();

		std::thread rpc_job(&DistManager::handle_rpcs, this);
		rpc_job.detach();
	}

	virtual ~DistManager (void)
	{
		data_queue_.Shutdown();
	}

	/// Implementation of iDistManager
	void expose_node (teq::TensptrT tens) override
	{
		if (estd::has(owners_, tens.get()) &&
			false == owners_[tens.get()].expired())
		{
			return;
		}
		std::string id = boost::uuids::to_string(DistManager::uuid_gen_());
		shared_nodes_.insert({id, tens.get()});
		owners_.emplace(tens.get(), tens);
		consul_.set_kv(node_lookup_prefix + id, consul_.id_);
	}

	/// Implementation of iDistManager
	std::optional<std::string> lookup_id (teq::iTensor* tens) const override
	{
		std::optional<std::string> out;
		if (estd::has(shared_nodes_.right, tens))
		{
			out = shared_nodes_.right.at(tens);
		}
		return out;
	}

	/// Implementation of iDistManager
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

		auto ref = remote_find_nodes(err, peer_id, id);
		if (nullptr == ref)
		{
			return nullptr;
		}

		shared_nodes_.insert({ref->node_id(), ref.get()});
		owners_.emplace(ref.get(), ref);
		remotes_.emplace(ref);
		return ref;
	}

	/// Implementation of iDistManager
	std::future<void> remote_evaluate (
		const std::string& peer_id,
		const estd::StrSetT& node_ids) override
	{
		if (false == estd::has(clients_, peer_id))
		{
			teq::fatalf("cannot find client %s", peer_id.c_str());
		}
		if (std::any_of(node_ids.begin(), node_ids.end(),
			[this](const std::string& id)
			{
				return false == estd::has(this->shared_nodes_.left, id);
			}))
		{
			teq::fatalf("some nodes from %s not found locally",
				peer_id.c_str());
		}

		GetDataRequest req;
		google::protobuf::RepeatedPtrField<std::string>
		uuids(node_ids.begin(), node_ids.end());
		req.mutable_uuids()->Swap(&uuids);
		return clients_.at(peer_id)->get_data(
			data_queue_, req, shared_nodes_);
	}

	/// Implementation of iDistManager
	teq::TensSetT find_reachable (
		error::ErrptrT& err,
		const teq::TensSetT& srcs,
		const estd::StrSetT& dests) override
	{
		teq::TensSetT local_target;
		for (auto dest : dests)
		{
			error::ErrptrT e;
			if (auto local = lookup_node(e, dest, false))
			{
				local_target.emplace(local.get());
			}
		}

		teq::TensSetT reached;
		estd::StrMapT<estd::StrSetT> remotes;
		estd::StrMapT<teq::TensSetT> refsrcs;
		for (auto src : srcs)
		{
			bool is_reachable = false;
			teq::LambdaVisit vis(
				[&](teq::iLeaf& leaf)
				{
					if (estd::has(local_target, &leaf))
					{
						is_reachable = true;
					}
				},
				[&](teq::iTraveler& trav, teq::iFunctor& func)
				{
					if (is_reachable || estd::has(local_target, &func))
					{
						is_reachable = true;
						return;
					}
					teq::multi_visit(trav, func.get_dependencies());
				});
			src->accept(vis);
			if (is_reachable)
			{
				reached.emplace(src);
			}
			else
			{
				auto refs = distr::reachable_refs(teq::TensT{src});
				for (auto ref : refs)
				{
					auto cid = ref->cluster_id();
					auto nid = ref->node_id();
					remotes[cid].emplace(nid);
					refsrcs[nid].emplace(src);
				}
			}
		}
		if (remotes.size() > 0)
		{
			for (auto& remote : remotes)
			{
				auto reachables = remote_find_reachable(err,
					remote.first, remote.second, dests);
				if (nullptr != err)
				{
					return {};
				}
				for (auto reachable : reachables)
				{
					auto& rsrcs = refsrcs[reachable];
					reached.insert(rsrcs.begin(), rsrcs.end());
				}
			}
		}
		return reached;
	}

	/// Implementation of iDistManager
	std::string get_id (void) const override
	{
		return consul_.id_;
	}

	/// Implementation of iDistManager
	DRefptrSetT get_remotes (void) const override
	{
		return remotes_;
	}

protected:
	DRefptrT remote_find_nodes (
		error::ErrptrT& err,
		const std::string& peer_id,
		const std::string& id)
	{
		FindNodesRequest req;
		FindNodesResponse res;
		req.add_uuids(id);
		auto status = clients_.at(peer_id)->find_nodes(req, res);
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
		return std::make_shared<DistRef>(
			egen::get_type(node.dtype()),
			teq::Shape(sdims),
			node.instance(), node_id);
	}

	estd::StrSetT remote_find_reachable (
		error::ErrptrT& err,
		const std::string& peer_id,
		const estd::StrSetT& srcs,
		const estd::StrSetT& dests)
	{
		if (false == estd::has(clients_, peer_id))
		{
			err = error::errorf(
				"cannot find client %s", peer_id.c_str());
			return {};
		}
		if (std::any_of(srcs.begin(), srcs.end(),
			[this](const std::string& id)
			{
				return false == estd::has(this->shared_nodes_.left, id);
			}))
		{
			err = error::errorf(
				"some nodes from %s not found locally",
				peer_id.c_str());
			return {};
		}

		FindReachableRequest req;
		FindReachableResponse res;
		google::protobuf::RepeatedPtrField<std::string>
		src_uuids(srcs.begin(), srcs.end());
		google::protobuf::RepeatedPtrField<std::string>
		dest_uuids(dests.begin(), dests.end());
		req.mutable_srcs()->Swap(&src_uuids);
		req.mutable_dests()->Swap(&dest_uuids);
		auto status = clients_.at(peer_id)->find_reachable(req, res);
		if (false == status.ok())
		{
			err = error::errorf("grpc status not ok: %s ()",
				status.error_message().c_str());
			return {};
		}

		auto& res_src = res.srcs();
		return estd::StrSetT(res_src.begin(), res_src.end());
	}

private:
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

using DistMgrptrT = std::shared_ptr<DistManager>;

}

#endif // DISTRIB_MANAGER_HPP
