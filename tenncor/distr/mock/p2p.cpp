#include "tenncor/distr/mock/p2p.hpp"

#ifdef DISTR_MOCK_P2P_HPP

error::ErrptrT check_health (const std::string& address,
	const std::string& health_id, size_t nretry)
{
	auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
	auto stub = distr::health::DistrHealth::NewStub(channel);
	error::ErrptrT err = nullptr;
	for (size_t i = 0; i < nretry; ++i)
	{
		grpc::ClientContext ctx;
		distr::health::CheckHealthRequest req;
		distr::health::CheckHealthResponse res;
		auto status = stub->CheckHealth(&ctx, req, &res);
		if (status.ok())
		{
			auto got_id = res.uuid();
			global::infof("health check id: %s", got_id.c_str());
			if (health_id != got_id)
			{
				err = error::errorf(
					"health check uuid %s does not match expected %s",
					got_id.c_str(), health_id.c_str());
			}
			else
			{
				return nullptr;
			}
		}
		else
		{
			err =  error::errorf("health check error: [%d] %s",
				status.error_code(), status.error_message().c_str());
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	}
	return err;
}

#endif
