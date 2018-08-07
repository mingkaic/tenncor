#include "glass/graph.hpp"

#include "soil/functor.hpp"
#include "soil/constant.hpp"
#include "soil/variable.hpp"

#ifdef GLASS_GRAPH_HPP

void save_graph (tenncor::GraphPb& out, const Session& in)
{
	for (auto npair : in.nodes_)
	{
		if (false == npair.second.expired())
		{
			out.add_labels(npair.first);
		}
	}
	out.set_id(in.hash());
	std::unordered_map<iNode*,uint32_t> visited;
	std::list<iNode*> nodes = order_nodes(in, visited);
	// serialize according to information gathered
	for (iNode* node : nodes)
	{
		tenncor::NodePb* nodepb = out.add_nodes();
		if (Functor* f = dynamic_cast<Functor*>(node))
		{
			auto args = f->get_refs();
			tenncor::FunctorPb func;
			auto argspb = func.mutable_args();
			std::transform(args.begin(), args.end(), argspb->begin(),
				[&visited](iNode* arg)
				{
					return visited[arg]; // assert arg in visited
				});
			func.set_encoding(std::string(1, (char) f->get_opcode()) +
				std::string(f->get_meta()));
			nodepb->set_type(tenncor::NodePb::FUNCTOR);
			nodepb->mutable_detail()->PackFrom(func);
		}
		else if (dynamic_cast<Constant*>(node))
		{
			tenncor::DataPb constpb;
			save_node(constpb, node);
			nodepb->set_type(tenncor::NodePb::CONSTANT);
			nodepb->mutable_detail()->PackFrom(constpb);
		}
		else if (dynamic_cast<Variable*>(node))
		{
			tenncor::DataInfoPb varpb;
			save_info(varpb, node);
			nodepb->set_type(tenncor::NodePb::VARIABLE);
			nodepb->mutable_detail()->PackFrom(varpb);
		}
		else
		{
			handle_error("serializing unsupported inode implementation");
		}
	}
}

struct IRNode
{
	void build (std::unordered_map<uint32_t,Nodeptr>& memo)
	{
		if (memo.end() != memo.find(idx_))
		{
			return;
		}
		switch (node_->type())
		{
			case tenncor::NodePb::FUNCTOR:
			{
				std::vector<Nodeptr> args;
				for (IRNode* child : children_)
				{
					child->build(memo);
					auto it = memo.find(child->idx_);
					assert(memo.end() != it);
					args.push_back(it->second);
				}
				tenncor::FunctorPb func;
				node_->detail().UnpackTo(&func);
				std::string encoding = func.encoding();
				OPCODE opcode = (OPCODE) encoding[0];
				std::shared_ptr<iPreOperator> preop =
					decode_meta(encoding.substr(1));
				memo.insert({idx_, Functor::get(args, *preop, opcode)});
			}
			break;
			case tenncor::NodePb::VARIABLE:
			{
				tenncor::DataInfoPb varpb;
				node_->detail().UnpackTo(&varpb);
				Shape shape;
				DTYPE type = load_info(shape, varpb);
				memo.insert({idx_, Variable::get(shape, type)});
			}
			break;
			case tenncor::NodePb::CONSTANT:
			{
				tenncor::DataPb constpb;
				node_->detail().UnpackTo(&constpb);
				Shape shape;
				std::string data;
				DTYPE type = load_node(data, shape, constpb);
				memo.insert({idx_, Constant::get(&data[0], shape, type)});
			}
			break;
			default:
				handle_error("deserializing invalid node type");
		}
	}

	uint32_t idx_;
	tenncor::NodePb* node_ = nullptr;
	std::vector<IRNode*> children_;
};

std::vector<Nodeptr> load_graph (Session& out, const tenncor::GraphPb& in)
{
	auto nodes = in.nodes();
	auto labels = in.labels();
	size_t nlabels = labels.size();
	size_t nnodes = nodes.size();
	std::vector<IRNode> irs(nnodes);
	for (size_t i = 0; i < nnodes; ++i)
	{
		if (tenncor::NodePb::FUNCTOR == nodes[i].type())
		{
			tenncor::FunctorPb func;
			nodes[i].detail().UnpackTo(&func);
			auto args = func.args();
			for (uint32_t j : args)
			{
				irs[i].children_.push_back(&irs[j]);
			}
		}
		irs[i].node_ = &nodes[i];
		irs[i].idx_ = i;
	}
	std::unordered_map<uint32_t,Nodeptr> outs;
	for (size_t i = 0; i < nnodes; ++i)
	{
		irs[i].build(outs);
	}
	// add label
	std::vector<Nodeptr> outvec;
	for (size_t i = 0; i < nlabels; ++i)
	{
		auto it = outs.find(i);
		assert(outs.end() != it);
		out.add(labels[i], it->second);
		outvec.push_back(it->second);
	}
	return outvec;
}

#endif
