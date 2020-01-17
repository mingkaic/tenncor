#include "teq/isession.hpp"

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
	PluginSession (teq::iDevice& device) : teq::Session(device) {}

	std::vector<PluginRefT> plugins_;

private:
	void process_reqs (teq::FuncListT& reqs) override
	{
		for (iPlugin& plugin : plugins_)
		{
			plugin.process(this->tracked_, reqs);
		}
	}
};

}

#endif // DBG_PLUGIN_SESS_HPP
