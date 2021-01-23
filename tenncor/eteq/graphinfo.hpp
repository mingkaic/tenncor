
#ifndef ETEQ_GRAPHINFO_HPP
#define ETEQ_GRAPHINFO_HPP

#include <boost/bimap.hpp>

#include "internal/teq/teq.hpp"

namespace eteq
{

struct GraphInfo;

// Tracks TensptrT with an Id
struct TensIdentity final
{
	static std::shared_ptr<TensIdentity> build (teq::TensptrT tens, const global::CfgMapptrT& ctx);
	
	~TensIdentity (void);

	TensIdentity (const TensIdentity& other) = delete;

	TensIdentity (TensIdentity&& other) = delete;

	TensIdentity& operator = (const TensIdentity& other) = delete;

	TensIdentity& operator = (TensIdentity&& other) = delete;

	teq::TensptrT get_tensor (void);
	
	std::string get_id (void) const;

private:
	TensIdentity (const std::string& tens, const global::CfgMapptrT& ctx);

	void cleanup (void);

	global::CfgMaprefT ctx_;

	GraphInfo* graphinfo_ = nullptr;

	std::string id_;
};

using TensIdptrT = std::shared_ptr<TensIdentity>;

using TensIdrefT = std::weak_ptr<TensIdentity>;

using IdRegistryT = boost::bimap<teq::TensptrT,TensIdentity*>;

// Defaults to identifying tensors by order of creation (obvious downside: new tensors created before existing model messes up identification at runtime)
struct GraphInfo final
{
	std::string build_id (void)
	{
		// linearly search for the next available id
		auto id = fmts::to_string(global_id_++);
		while (estd::has(idrefs_, id))
		{
			id = fmts::to_string(global_id_++);
		}
		return id;
	}

	void emplace (const teq::TensptrT& tens, const TensIdptrT& id)
	{
		tens_ids_.insert({tens, id.get()});
		idrefs_.emplace(id->get_id(), id);
	}
	
	teq::TensptrT get (const std::string& id) const
	{
		auto idref = estd::try_get(idrefs_, id, TensIdrefT());
		if (idref.expired())
		{
			return nullptr;
		}
		return get(idref.lock().get());
	}

	teq::TensptrT get (TensIdentity* id) const
	{
		if (estd::has(tens_ids_.right, id))
		{
			return tens_ids_.right.at(id);
		}
		return nullptr;
	}

	TensIdptrT get (const teq::TensptrT& tens) const
	{
		if (estd::has(tens_ids_.left, tens))
		{
			auto id = idrefs_.at(tens_ids_.left.at(tens)->get_id());
			if (false == id.expired())
			{
				return id.lock();
			}
		}
		return nullptr;
	}

	void erase (TensIdentity* id)
	{
		if (estd::has(tens_ids_.right, id))
		{
			tens_ids_.right.erase(id);
			idrefs_.erase(id->get_id());
		}
	}

	void erase (const teq::TensptrT& tens)
	{
		if (estd::has(tens_ids_.left, tens))
		{
			auto id = get(tens);
			tens_ids_.left.erase(tens);
			idrefs_.erase(id->get_id());
		}
	}

	void replace (const teq::TensptrT& src, const teq::TensptrT& dst)
	{
		if (auto id = get(src))
		{
			tens_ids_.left.erase(src);
			tens_ids_.insert({dst, id.get()});
		}
	}

	void foreach (std::function<void(teq::TensptrT,const std::string&)> cb) const
	{
		for (auto tid : tens_ids_)
		{
			cb(tid.left, tid.right->get_id());
		}
	}

private:
	size_t global_id_ = 1000; // default id used for saving/loading ETensors

	IdRegistryT tens_ids_;

	types::StrUMapT<TensIdrefT> idrefs_;
};

void set_graphinfo (GraphInfo* gi, global::CfgMapptrT ctx = global::context());

GraphInfo& get_graphinfo (const global::CfgMapptrT& ctx = global::context());

}

#endif // ETEQ_GRAPHINFO_HPP
