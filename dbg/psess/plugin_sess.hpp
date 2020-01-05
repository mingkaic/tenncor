#include "teq/session.hpp"

#ifndef DBG_PLUGIN_SESS_HPP
#define DBG_PLUGIN_SESS_HPP

namespace dbg
{

struct iPlugin
{
	virtual ~iPlugin (void) = default;

	virtual void process (
		teq::TensptrSetT& tracked, teq::FuncListT& targets)= 0;
};

using PluginRefT = std::reference_wrapper<iPlugin>;

struct PluginSession final : public teq::Session
{
	std::vector<PluginRefT> plugins_;

private:
	void calc_reqfuncs (teq::FuncListT& reqs) override
	{
		for (auto& op : reqs)
		{
			op->calc();
		}
		for (iPlugin& plugin : plugins_)
		{
			plugin.process(tracked_, reqs);
		}
	}
};

}

#endif // DBG_PLUGIN_SESS_HPP
