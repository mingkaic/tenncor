#include <boost/bimap.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/lexical_cast.hpp>

#include "experimental/distrib/client.hpp"
#include "experimental/distrib/server.hpp"

#ifndef DISTRIB_SESSION_HPP
#define DISTRIB_SESSION_HPP

namespace distrib
{

using UuidSetT = std::unordered_set<std::string>;

struct DataResponse final
{
	grpc::Status status_;

	distr::NodeData res_;
};

struct DistribSess final : public iDistribSess
{
	/// UUID random generator
	static boost::uuids::random_generator uuid_gen_;

	DistribSess (const std::string& address,
		const std::string& peer = "",
		const ClientConfig& cfg = ClientConfig()) :
		service_(this), address_(address)
	{
		grpc::ServerBuilder builder;
		builder.AddListeningPort(address, grpc::InsecureServerCredentials());
		builder.RegisterService(&service_);
		server_ = builder.BuildAndStart();
		teq::infof("server listening on %s", address.c_str());

		if (peer.empty())
		{
			return;
		}
		clients_.insert({peer, std::make_unique<DistrCli>(peer, cfg)});
		auto clients = clients_[peer]->list_instances();
		clients_[peer]->add_instance(address);
		for (auto client : clients)
		{
			clients_.insert({client, std::make_unique<DistrCli>(client, cfg)});
			clients_[client]->add_instance(address);
		}
	}

	~DistribSess (void)
	{
		server_->Shutdown();
	}

	/// Implementation of iDistribSess
	void call (const DRefsT& refs) override
	{
		std::unordered_map<std::string,UuidSetT> clusters;
		for (auto& ref : refs)
		{
			auto cid = ref->cluster_id();
			clusters[cid].emplace(lookup_id(ref));
		}
		for (auto& cluster : clusters)
		{
			data_request(cluster.first, cluster.second);
		}
	}

	/// Implementation of iDistribSess
	void sync (void) override
	{
		void* got_tag;
		bool ok = false;
		while (data_queue_.Next(&got_tag, &ok))
		{
			auto handler = static_cast<iStreamHandler<distr::NodeData>*>(got_tag);
			if (ok)
			{
				handler->handle_resp();
				auto& res = handler->get_data();
				auto uuid = res.uuid();
				auto ref = static_cast<iDistRef*>(shared_nodes_.left.at(uuid).get());
				ref->update_data(res.data().data(), res.version());
			}
			else
			{
				auto status = handler->done();
				if (status.ok())
				{
					teq::infof("server response completed: %p", handler);
				}
				else
				{
					teq::warnf("server response failed: %p", handler);
				}
				delete handler;
			}
		}
	}

	std::string lookup_id (teq::TensptrT tens) const override
	{
		return estd::must_getf(shared_nodes_.right, tens,
			"cannot find tensor %s (%p)",
			tens->to_string().c_str(), tens.get());
	}

	teq::TensptrT lookup_node (const std::string& id, bool recursive = true) override
	{
		if (false == estd::has(shared_nodes_.left, id))
		{
			if (false == recursive)
			{
				return nullptr;
			}
			// search for remote references
			distr::NodeMeta node;
			if (false == lookup_request(node, id))
			{
				teq::fatalf("failed to find node %s", id.c_str());
			}
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

	std::string get_address (void) const override
	{
		return address_;
	}

private:
	void store_tracked (const teq::TensptrSetT& locals) override
	{
		for (teq::TensptrT local : locals)
		{
			shared_nodes_.insert({boost::uuids::to_string(DistribSess::uuid_gen_()), local});
		}
	}

	/// Return the cluster id that owns node_id
	bool lookup_request (distr::NodeMeta& node,
		const std::string& node_id)
	{
		grpc::CompletionQueue cq;
		distr::FindNodesRequest req;
		req.add_uuids(node_id);

		std::vector<iResponseHandler<distr::FindNodesResponse>*> responses;
		for (auto& inst : clients_)
		{
			responses.push_back(inst.second->lookup_node(cq, req));
		}

		void* got_tag;
		bool ok = false;
		bool found = false;
		while (cq.Next(&got_tag, &ok))
		{
			auto handler = static_cast<iResponseHandler<distr::FindNodesResponse>*>(got_tag);
			if (ok)
			{
				auto& res = handler->get_response();
				if (res.values().size() > 0)
				{
					node = res.values().at(0);
					found = true;
					break;
				}
			}
			else
			{
				auto status = handler->check_status();
				if (status.ok())
				{
					teq::infof("server response completed: %p", handler);
				}
				else
				{
					teq::fatalf("server response failed: %p", handler);
				}
			}
		}
		for (auto res : responses)
		{
			delete res;
		}
		return found;
	}

	/// Make request to cluster_id for data updates to all nodes
	void data_request (const std::string& cluster_id, const UuidSetT& nodes)
	{
		distr::GetDataRequest req;
		for (const auto& node_id : nodes)
		{
			req.add_uuids(node_id);
		}
		clients_.at(cluster_id)->get_data(data_queue_, req);
	}

	boost::bimap<std::string,teq::TensptrT> shared_nodes_;

	std::unordered_map<std::string,DistrCliPtrT> clients_;

	std::unique_ptr<grpc::Server> server_;

	grpc::CompletionQueue data_queue_;

	DistrService service_;

	std::string address_;
};

}

#endif // DISTRIB_SESSION_HPP
