//
// Created by Ming Kai Chen on 2017-11-07.
//

#include "include/edgeinfo/rpc_record/grpc_record.hpp"

#if defined(RPC_RCD) && defined(grpc_record_HPP)

namespace rocnnet_record
{

std::unique_ptr<igraph_record> record_status::rec =
	std::make_unique<rpc_record>("localhost", 50981);
bool record_status::rec_good = true;

rpc_record::rpc_record (std::string host, size_t port) :
	node_cache_([](const visor::NodeMessage& msg) -> std::string
	{
		return msg.id();
	}),
	edge_cache_([](const visor::EdgeMessage& msg) -> std::string
	{
		return nnutils::formatter() << msg.idx() << ":"
			<< msg.obsid() << ":" << msg.subid();
	})
{
	std::string address = nnutils::formatter() << host << ":" << port;
	server_ = nnet::stoppable_thread(spawn_server, address,
		std::inode*(node_cache_), std::inode*(edge_cache_));
}

rpc_record::~rpc_record (void)
{
	server_.stop();
}

void rpc_record::node_release (const nnet::subject* sub)
{
	std::string sid = static_cast<const nnet::inode*>(sub)->get_uid();
	visor::NodeMessage message;
	message.set_id(sid);
	message.set_status(
		visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_REMOVE);

	node_cache_.add_latest(message);
}

void rpc_record::data_update (const nnet::subject* sub)
{
	std::string sid = static_cast<const nnet::inode*>(sub)->get_uid();
	visor::NodeMessage message;
	message.set_id(sid);
	message.set_status(
		visor::NodeMessage_NodeStatus::NodeMessage_NodeStatus_UPDATE);

	node_cache_.add_latest(message);
}

void rpc_record::edge_capture (const nnet::iobserver* obs,
	const nnet::subject* sub, size_t obs_idx)
{
	std::string oid = static_cast<const nnet::iconnector*>(obs)->get_uid();
	std::string sid = static_cast<const nnet::inode*>(sub)->get_uid();
	visor::EdgeMessage message;
	message.set_obsid(oid);
	message.set_subid(sid);
	message.set_idx(obs_idx);
	message.set_status(
		visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_ADD);

	edge_cache_.add_latest(message);
}

void rpc_record::edge_release (const nnet::iobserver* obs,
	const nnet::subject* sub, size_t obs_idx)
{
	std::string oid = static_cast<const nnet::iconnector*>(obs)->get_uid();
	std::string sid = static_cast<const nnet::inode*>(sub)->get_uid();
	visor::EdgeMessage message;
	message.set_obsid(oid);
	message.set_subid(sid);
	message.set_idx(obs_idx);
	message.set_status(
		visor::EdgeMessage_EdgeStatus::EdgeMessage_EdgeStatus_REMOVE);

	edge_cache_.add_latest(message);
}

}

#endif
