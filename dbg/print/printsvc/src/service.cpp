
#include "dbg/print/printsvc/service.hpp"

#ifdef DISTR_PRINT_SERVICE_HPP

namespace distr
{

namespace print
{

void DistrPrintService::print_ascii (std::ostream& os, teq::iTensor* tens)
{
	AsciiTemplate ascii(tens, printopts_);
	auto remotes = ascii.remotes_;
	for (size_t depth = 0; depth < depthlimit_ && remotes.size() > 0; ++depth)
	{
		remotes = print_ascii_remotes(remotes);
	}
	cache_.stitch_ascii(os, ascii, "", "");
}

bool DistrPrintService::process_list_ascii (const ListAsciiRequest& req,
	types::StringsT::iterator& it, AsciiEntry& reply)
{
	auto uuid = *it;

	error::ErrptrT err = nullptr;
	auto tens = iosvc_->lookup_node(err, uuid, false);
	if (nullptr != err)
	{
		global::errorf("[server %s] %s", get_peer_id().c_str(), err->to_string().c_str());
		return false;
	}

	reply.set_uuid(uuid);
	AsciiTemplate temp(tens.get(), printopts_);
	reply.set_format(temp.format_.str());
	for (auto& dep : temp.remotes_)
	{
		auto adep = reply.add_deps();
		adep->set_refid(dep.refid_);
		adep->set_clusterid(dep.clusterid_);
		adep->set_prefix(dep.prefix_);
	}
	return true;
}

AsciiRemotesT DistrPrintService::print_ascii_remotes (const AsciiRemotesT& remotes)
{
	AsciiRemotesT nexts;
	types::StrUMapT<types::StrUSetT> servers;
	for (auto& dep : remotes)
	{
		auto rid = dep.refid_;
		if (false == estd::has(cache_.remote_templates_, rid))
		{
			if (dep.clusterid_ == get_peer_id())
			{
				// process local
				cache_.remote_templates_.emplace(rid,
					AsciiTemplate(iosvc_->must_lookup_node(
						rid, false).get(), printopts_));
				auto& temp = cache_.remote_templates_.at(rid);
				nexts.insert(nexts.end(),
					temp.remotes_.begin(), temp.remotes_.end());
			}
			else
			{
				servers[dep.clusterid_].emplace(rid);
			}
		}
	}
	if (servers.empty())
	{
		return nexts;
	}
	std::list<egrpc::ErrPromiseptrT> completions;
	for (auto server : servers)
	{
		auto peer_id = server.first;
		auto& nodes = server.second;

		error::ErrptrT err = nullptr;
		auto client = get_client(err, peer_id);
		if (nullptr != err)
		{
			global::error(err->to_string());
			continue;
		}

		google::protobuf::RepeatedPtrField<std::string>
		node_ids(nodes.begin(), nodes.end());

		ListAsciiRequest req;
		req.mutable_uuids()->Swap(&node_ids);
		completions.push_back(client->list_ascii(*cq_, req,
			[&](AsciiEntry& res)
			{
				auto uuid = res.uuid();
				auto& deps = res.deps();
				std::vector<AsciiRemote> remotes;
				for (auto& dep : deps)
				{
					AsciiRemote rem{
						dep.refid(),
						dep.clusterid(),
						dep.prefix()
					};
					nexts.push_back(rem);
					remotes.push_back(rem);
				}
				cache_.remote_templates_.emplace(uuid,
					AsciiTemplate(res.format(), remotes));
			}));
	}
	egrpc::wait_for(completions,
	[](error::ErrptrT err)
	{
		global::fatal(err->to_string());
	});
	return nexts;
}

}

error::ErrptrT register_printsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("printsvc requires iosvc already registered");
	}
	svcs.add_entry<print::DistrPrintService>(print::printsvc_key,
		[&]{ return new print::DistrPrintService(cfg, iosvc); });
	return nullptr;
}

print::DistrPrintService& get_printsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(print::printsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			print::printsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<print::DistrPrintService&>(*svc);
}

}

#endif
