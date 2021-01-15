
#ifndef DBG_PLUGIN_EVAL_HPP
#define DBG_PLUGIN_EVAL_HPP

#include "internal/teq/teq.hpp"

namespace dbg
{

struct iPlugin
{
	virtual ~iPlugin (void) = default;

	virtual void process (
		const teq::TensSetT& targets,
		const teq::TensSetT& visited)= 0;
};

using PluginRefT = std::reference_wrapper<iPlugin>;

struct PlugableEvaluator final : public teq::iEvaluator
{
	void evaluate (
		teq::iDevice& device,
		const teq::TensSetT& targets,
		const teq::TensSetT& ignored = {}) override
	{
		teq::TravEvaluator eval(device, targets, ignored);
		teq::multi_visit(eval, targets);
		for (iPlugin& plugin : plugins_)
		{
			plugin.process(targets, eval.visited_);
		}
	}

	void add_plugin (iPlugin& plugin)
	{
		plugins_.push_back(plugin);
	}

	std::vector<PluginRefT> plugins_;
};

}

#endif // DBG_PLUGIN_EVAL_HPP
