
#include "egrpc/server_async.hpp"

#include "dbg/print/print.hpp"

#include "distrib/services/io/service.hpp"

#include "dbg/distr_ext/print/client.hpp"

#ifndef DISTRIB_PRINT_SERVICE_HPP
#define DISTRIB_PRINT_SERVICE_HPP

namespace distr
{

#define _ERR_CHECK(ERR, STATUS, ALIAS)\
if (nullptr != ERR)\
{\
	global::errorf("[server %s] %s", ALIAS,\
		ERR->to_string().c_str());\
	return grpc::Status(STATUS, ERR->to_string());\
}

const std::string printsvc_key = "dbg_printsvc";

struct AsciiRemote
{
	std::string refid_;

	std::string clusterid_;

	std::string prefix_;
};

using AsciiRemotesT = std::vector<AsciiRemote>;

const std::string ascii_formatkey = "{xj]yq<";

struct AsciiTemplate
{
	AsciiTemplate (std::string format, std::vector<AsciiRemote> remotes) :
		remotes_(remotes)
	{
		format_ << format;
	}

	AsciiTemplate (teq::iTensor* tens, bool showshape)
	{
		teq::TensptrsT dummies;
		PrettyTree<teq::iTensor*> renderer(
			[&](teq::iTensor*& root, size_t depth) -> teq::TensT
			{
				if (auto f = dynamic_cast<teq::iFunctor*>(root))
				{
					auto children = f->get_args();
					std::vector<teq::iTensor*> tens;
					tens.reserve(children.size());
					std::transform(children.begin(), children.end(),
						std::back_inserter(tens),
						[](teq::TensptrT child)
						{
							return child.get();
						});
					auto deps = f->get_dependencies();
					if (deps.size() > children.size())
					{
						auto dummy = std::make_shared<MockFunctor>(
							teq::TensptrsT(deps.begin() + children.size(), deps.end()),
							teq::Opcode{dummy_label, 0});
						tens.push_back(dummy.get());
						dummies.push_back(dummy);
					}
					return tens;
				}
				return {};
			},
			[&, this](std::ostream& out, teq::iTensor*& root, const std::string& prefix)
			{
				if (root)
				{
					if (auto ref = dynamic_cast<iDistrRef*>(root))
					{
						out << ascii_formatkey;
						remotes_.push_back(AsciiRemote{ref->node_id(), ref->cluster_id(), prefix});
						return;
					}
					out << "(";
					if (auto var = dynamic_cast<teq::iLeaf*>(root))
					{
						out << teq::get_usage_name(var->get_usage()) << ":";
					}
					out << root->to_string();
					if (showshape)
					{
						out << root->shape().to_string();
					}
					out << ")";
				}
			});
		renderer.node_wrap = {"", ""};
		renderer.print(format_, tens);
	}

	std::stringstream format_;

	std::vector<AsciiRemote> remotes_;
};

struct DistrPrintService final : public PeerService<DistrPrintCli>
{
	DistrPrintService (const PeerServiceConfig& cfg,
		DistrIOService* iosvc,
		bool showshape = false) :
		PeerService<DistrPrintCli>(cfg),
		iosvc_(iosvc), showshape_(showshape) {}

	void print_ascii (std::ostream& os, teq::iTensor* tens)
	{
		AsciiTemplate ascii(tens, showshape_);
		render(os, ascii);
	}

	void register_service (grpc::ServerBuilder& builder) override
	{
		builder.RegisterService(&service_);
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		// ListAscii
		auto lascii_logger = std::make_shared<global::FormatLogger>(
			&global::get_logger(), fmts::sprintf("[server %s:ListAscii] ",
				get_peer_id().c_str()));
		new egrpc::AsyncServerStreamCall<print::ListAsciiRequest,
			print::AsciiEntry,types::StringsT>(lascii_logger,
			[this](grpc::ServerContext* ctx, print::ListAsciiRequest* req,
				grpc::ServerAsyncWriter<print::AsciiEntry>* writer,
				grpc::CompletionQueue* cq, grpc::ServerCompletionQueue* ccq,
				void* tag)
			{
				this->service_.RequestListAscii(ctx, req, writer, cq, ccq, tag);
			},
			[this](types::StringsT& states, const print::ListAsciiRequest& req)
			{
				return this->startup_list_ascii(states, req);
			},
			[this](const print::ListAsciiRequest& req,
				types::StringsT::iterator& it, print::AsciiEntry& reply)
			{
				return this->process_list_ascii(req, it, reply);
			}, &cq);
	}

private:
	grpc::Status startup_list_ascii (
		types::StringsT& roots,
		const print::ListAsciiRequest& req)
	{
		auto& uuids = req.uuids();
		roots.insert(roots.end(), uuids.begin(), uuids.end());
		return grpc::Status::OK;
	}

	bool process_list_ascii (const print::ListAsciiRequest& req,
		types::StringsT::iterator& it,
		print::AsciiEntry& reply)
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
		AsciiTemplate temp(tens.get(), showshape_);
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

	void render (std::ostream& os, AsciiTemplate& ascii)
	{
		auto remotes = ascii.remotes_;
		for (size_t depth = 0; depth < depthlimit_ && remotes.size() > 0; ++depth)
		{
			remotes = resolve_remotes(remotes);
		}
		render_helper(os, ascii, "", "");
	}

	void render_helper (std::ostream& os, AsciiTemplate& ascii,
		const std::string& prefix, const std::string& first_line_prefix)
	{
		size_t i = 0, n = ascii.remotes_.size();
		auto format = ascii.format_.str();
		fmts::trim(format);
		auto lines = fmts::split(format, "\n");
		for (size_t lno = 0, nl = lines.size(); lno < nl; ++lno)
		{
			auto line = lines[lno];
			if (i < n)
			{
				auto formbegin = line.find(ascii_formatkey);
				if (formbegin != std::string::npos)
				{
					auto it = line.begin();
					std::string init(it, it + formbegin);
					auto& remote = ascii.remotes_[i];
					++i;
					if (estd::has(remote_templates_, remote.refid_))
					{
						render_helper(os,
							remote_templates_.at(remote.refid_),
							prefix + remote.prefix_,
							prefix + init + "[" + remote.clusterid_ + "]:");
						continue;
					}
					else
					{
						line = "(?)";
					}
				}
			}
			// add special prefix for first line only
			if (lno == 0)
			{
				os << first_line_prefix;
			}
			else
			{
				os << prefix;
			}
			os << line << "\n";
		}
	}

	AsciiRemotesT resolve_remotes (const AsciiRemotesT& remotes)
	{
		AsciiRemotesT nexts;
		types::StrUMapT<types::StrUSetT> servers;
		for (auto& dep : remotes)
		{
			auto rid = dep.refid_;
			if (false == estd::has(remote_templates_, rid))
			{
				if (dep.clusterid_ == get_peer_id())
				{
					// process local
					remote_templates_.emplace(rid,
						AsciiTemplate(iosvc_->must_lookup_node(
							rid, false).get(), showshape_));
					auto& temp = remote_templates_.at(rid);
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
		std::vector<egrpc::ErrFutureT> completions;
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

			print::ListAsciiRequest req;
			req.mutable_uuids()->Swap(&node_ids);
			completions.push_back(client->list_ascii(cq_, req,
				[&](print::AsciiEntry& res)
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
					remote_templates_.emplace(uuid,
						AsciiTemplate(res.format(), remotes));
				}));
		}
		for (auto& done : completions)
		{
			while (done.valid() && done.wait_for(
				std::chrono::milliseconds(1)) ==
				std::future_status::timeout);
			if (auto err = done.get())
			{
				global::fatal(err->to_string());
			}
		}
		return nexts;
	}

	types::StrUMapT<AsciiTemplate> remote_templates_;

	DistrIOService* iosvc_;

	bool showshape_;

	size_t depthlimit_ = 10;

	print::DistrPrint::AsyncService service_;
};

#undef _ERR_CHECK

error::ErrptrT register_printsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg);

DistrPrintService& get_printsvc (iDistrManager& manager);

}

#endif // DISTRIB_PRINT_SERVICE_HPP
