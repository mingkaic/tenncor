#ifndef EIGEN_CONFIG_HPP
#define EIGEN_CONFIG_HPP

#include "teq/config.hpp"

#include "eigen/device.hpp"
#include "eigen/random.hpp"

namespace eigen
{

const std::string device_key = "device";

const std::string rengine_key = "rengine";

struct Config final : public estd::iConfig
{
	std::vector<string> get_names (void) const override
	{
		return {
			teq::logger_key,
			eigen::device_key,
			eigen::rengine_key,
		};
	}

	void* get_obj (std::string cfg_name) const override
	{
		if (cfg_name == teq::logger_key)
		{
			return &logger_;
		}
		else if (cfg_name == eigen::device_key)
		{
			return &device_;
		}
		else if (cfg_name == eigen::rengine_key)
		{
			return &engine_;
		}
		logger_.error(fmts::sprintf(
			"failed to find config name %s", cfg_name.c_str()));
		return nullptr;
	}

	teq::DefLogger logger_;

	eigen::Device device_;

	eigen::EngineT engine_;
};

static bool registered = []{ teq::global_config = std::make_shared<Config>(); };

}

#endif // EIGEN_CONFIG_HPP
