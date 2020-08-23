
#ifndef DISTRIB_IO_SERVICE_HPP
#define DISTRIB_IO_SERVICE_HPP

#include "tenncor/distrib/imanager.hpp"
#include "tenncor/distrib/iosvc/client.hpp"
#include "tenncor/distrib/iosvc/data.hpp"
#include "tenncor/distrib/iosvc/pb_helper.hpp"

namespace distr
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

const std::string iosvc_key = "distr_iosvc";

struct DistrIOService final : public PeerService<DistrIOCli>
{
	DistrIOService (const PeerServiceConfig& cfg) :
		PeerService<DistrIOCli>(cfg), data_(cfg.consul_) {}

	std::string expose_node (teq::TensptrT tens)
	{
		return data_.cache_tens(tens);
	}

	std::optional<std::string> lookup_id (teq::iTensor* tens) const
	{
		return data_.get_id(tens);
	}

	teq::TensptrT lookup_node (error::ErrptrT& err,
		const std::string& id, bool recursive = true)
	{
		if (auto out = data_.get_tens(id))
		{
			return out;
		}
		if (false == recursive)
		{
			err = error::errorf(
				"no id %s found: will not recurse", id.c_str());
			return nullptr;
		}

		auto peer_id = data_.get_peer(id);
		if (false == bool(peer_id))
		{
			err = error::errorf("no peer found for node %s", id.c_str());
			return nullptr;
		}

		auto client = get_client(err, *peer_id);
		if (nullptr != err)
		{
			return nullptr;
		}

		io::ListNodesRequest req;
		DRefptrT ref = nullptr;
		req.add_uuids(id);
		auto promise = client->list_nodes(cq_, req,
			[&, this](io::ListNodesResponse& res)
			{
				if (res.values().empty())
				{
					err = error::errorf("no result found in received peer '%s'",
						peer_id->c_str());
					return;
				}
				auto node = res.values().at(0);
				ref = node_meta_to_ref(node);
				this->data_.cache_tens(ref);
			});
		auto done = promise->get_future();
		wait_on_future(done);
		return ref;
	}

	teq::TensptrT must_lookup_node (
		const std::string& id,
		bool recursive = true)
	{
		error::ErrptrT err = nullptr;
		auto out = lookup_node(err, id, recursive);
		if (nullptr != err)
		{
			global::fatal(err->to_string());
		}
		return out;
	}

	std::string must_lookup_id (teq::iTensor* tens)
	{
		auto id = lookup_id(tens);
		if (false != bool(id))
		{
			global::fatalf("failed to find '%s'[%p]", tens->to_string().c_str(), tens);
		}
		return *id;
	}

	DRefptrSetT get_remotes (void) const
	{
		return data_.get_remotes();
 	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// ListNodes
		auto lnodes_logger = std::make_shared<global::FormatLogger>(&global::get_logger(),
			fmts::sprintf("[server %s:ListNodes] ", get_peer_id().c_str()));
		new egrpc::AsyncServerCall<io::ListNodesRequest,
			io::ListNodesResponse>(lnodes_logger,
			[this](grpc::ServerContext* ctx, io::ListNodesRequest* req,
				grpc::ServerAsyncResponseWriter<io::ListNodesResponse>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestListNodes(ctx, req, writer, cq, ccq, tag);
			},
			[this](
				const io::ListNodesRequest& req,
				io::ListNodesResponse& res)
			{
				return this->list_nodes(req, res);
			}, &cq);
	}

private:
	grpc::Status list_nodes (
		const io::ListNodesRequest& req,
		io::ListNodesResponse& res)
	{
		auto alias = fmts::sprintf("%s:ListNodes", get_peer_id().c_str());
		auto& uuids = req.uuids();
		for (const std::string& uuid : uuids)
		{
			error::ErrptrT err = nullptr;
			auto tens = lookup_node(err, uuid, false);
			_ERR_CHECK(err, grpc::NOT_FOUND, alias.c_str());

			io::NodeMeta* out = res.add_values();
			tens_to_node_meta(*out, get_peer_id(), uuid, tens);
		}
		return grpc::Status::OK;
	}

	io::DistrInOut::AsyncService service_;

	DistrIOData data_;
};

#undef _ERR_CHECK

error::ErrptrT register_iosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

DistrIOService& get_iosvc (iDistrManager& manager);

}

#endif // DISTRIB_IO_SERVICE_HPP
