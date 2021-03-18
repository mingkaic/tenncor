#include "dbg/profile/graph.hpp"

#ifdef DBG_PROFILE_GRAPH_HPP

namespace dbg
{

namespace profile
{

void remote_profile (const std::string& addr, eteq::ETensorsT roots)
{
	std::string host = addr + "/graph";
	try
	{
		auto channel = grpc::CreateChannel(host,
			grpc::InsecureChannelCredentials());
		TenncorProfileClient client(channel, egrpc::ClientConfig{});

		tenncor_profile::CreateProfileRequest req;
		onnx::ModelProto* pb_model = req.mutable_model();

		eigen::Device realdev(
			eigen::get_runtime(), std::numeric_limits<size_t>::max());
		ProfilerDevice device(realdev);

		teq::TensSetT rootset;
		teq::multi_get(roots.begin(), roots.end(),
			std::inserter(rootset, rootset.end()));
		teq::Evaluator().evaluate(device, rootset);
		auto owners = teq::track_ownptrs(roots);

		auto gen = global::get_generator();
		onnx::TensptrIdT ids;
		auto rt = req.mutable_runtime();
		for (auto stat : device.stats_)
		{
			auto key = stat.first;
			auto val = stat.second;
			auto uuid = gen->get_str();
			rt->insert({uuid, (int64_t) val});
			ids.insert({owners.at(key), uuid});
		}
		tcr::save_model(*pb_model, roots, ids);

		grpc::Status status = client.create_profile(req);

		if (status.ok())
		{
			global::infof("successful profile creation to `%s`", host.c_str());
		}
		else
		{
			global::errorf("failed profile creation to `%s`", host.c_str());
		}
	}
	catch (const std::exception& e)
	{
		global::errorf("failed to estabilish connection to `%s`: %s",
			host.c_str(), e.what());
	}
}

}

}

#endif
