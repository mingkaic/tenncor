#include "ead/constant.hpp"
#include "ead/functor.hpp"

#include "ead/dbg/interactive_client.hpp"

#ifndef EAD_DBG_HPP
#define EAD_DBG_HPP

namespace ead
{

static const std::string default_server = "localhost:50051";

const std::string default_label = "normal_graph";

const std::string continue_msg = "continue (y/N)? >";

const std::string invalid_break_signal_msg =
	"accepts y/n/yes/no (case insensitive) as response";

struct Painter : public ade::iTraveler
{
	/// Implementation of iTraveler
	void visit (ade::iLeaf* leaf) override
	{
		if (visited_.end() == visited_.find(leaf) &&
			(canvas_.end() == canvas_.find(leaf)))
		{
			visited_.emplace(leaf);
			canvas_.emplace(leaf, NodeMetadata{false, {},
				std::weak_ptr<ade::iTensor>()});
		}
	}

	/// Implementation of iTraveler
	void visit (ade::iFunctor* func) override
	{
		if (visited_.end() == visited_.find(func) &&
			(canvas_.end() == canvas_.find(func)))
		{
			visited_.emplace(func);
			canvas_.emplace(func, NodeMetadata{true, {},
				std::weak_ptr<ade::iTensor>()});
			auto children = func->get_children();
			for (auto& child : children)
			{
				auto ctens = child.get_tensor();
				ctens->accept(*this);
				canvas_[ctens.get()].ownership_ = ctens;
			}
		}
	}

	std::unordered_set<ade::iTensor*> visited_;

	GraphCanvasT canvas_;
};

template <typename T>
struct InteractiveDebugger
{
	void track (NodeptrT<T>& node, std::string label = default_label)
	{
		Painter painter;
		auto root = node->get_tensor();
		root->accept(painter);
		for (auto& canvas_info : painter.canvas_)
		{
			auto tens = canvas_info.first;
			auto it = graph_canvas_.find(tens);
			if (graph_canvas_.end() == it)
			{
				canvas_info.second.labels_ = {label};
				graph_canvas_.emplace(tens, canvas_info.second);
			}
			else
			{
				graph_canvas_[tens].labels_.emplace(label);
			}
		}
		graph_canvas_[root.get()].ownership_ = root;
	}

	// update display then wait signal before continue (single-thread blocker)
	void set_break (void)
	{
		// cleanup canvas
		clean_up();

		// update display
		GraphClient client(default_server);
		EdgesT active_edges;
		active_edges.reserve(edges_.size());
		auto et = graph_canvas_.end();
		std::copy_if(edges_.begin(), edges_.end(),
			std::back_inserter(active_edges),
			[&](Edge& edge)
			{
				if (edge.expired())
				{
					return false;
				}
				auto parent = edge.parent_.lock().get();
				auto child = edge.child_.lock().get();
				return et == graph_canvas_.find(parent) &&
					et == graph_canvas_.find(child);
			});
		client.send(graph_canvas_, active_edges);

		// handle signal
		// todo: make more elegant and flexible
		std::string break_signal;
		while (!std::cin.eof())
		{
			std::cout << continue_msg;
			std::getline(std::cin, break_signal);
			// parse break_signal
			std::transform(break_signal.begin(), break_signal.end(),
				break_signal.begin(), [](char c) { return std::tolower(c); });
			if (break_signal == "y" || break_signal == "yes")
			{
				return;
			}
			else if (break_signal != "n" && break_signal != "no")
			{
				logs::warn(invalid_break_signal_msg);
			}
			break_signal = "";
		}
	}

	void clean_up (void)
	{
		for (auto it = graph_canvas_.begin(), et = graph_canvas_.end(); it != et;)
		{
			auto& owner = it->second.ownership_;
			if (owner.expired())
			{
				it = graph_canvas_.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	GraphCanvasT graph_canvas_;

	EdgesT edges_;
};

}

#endif // EAD_DBG_HPP
