#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "ead/generated/pyapi.hpp"
#include "ead/ead.hpp"
#include "ead/parse.hpp"

#include "dbg/grpc/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(grpc_dbg, m)
{
	m.doc() = "dbg ade equation graphs using interactive grpc session";

	py::object isess = (py::object)
		py::module::import("ead.ead").attr("iSession");

	py::class_<dbg::InteractiveSession,
		std::shared_ptr<dbg::InteractiveSession>> session(
			m, "InteractiveSession", isess);
	py::implicitly_convertible<ead::iSession,dbg::InteractiveSession>();

	m.def("get_isess",
		[](std::string host, size_t request_duration, size_t stream_duration)
		{
			return std::make_shared<dbg::InteractiveSession>(host,
				dbg::ClientConfig{
					std::chrono::milliseconds(request_duration),
					std::chrono::milliseconds(stream_duration),
				});
		}, py::arg("host") = "localhost:50051",
			py::arg("request_dur") = 1000,
			py::arg("stream_dur") = 30000);
	session
		.def("track",
		[](py::object self, ead::NodesT<PybindT> roots)
		{
			auto sess = self.cast<dbg::InteractiveSession*>();
			ade::TensT troots;
			troots.reserve(roots.size());
			std::transform(roots.begin(), roots.end(),
				std::back_inserter(troots),
				[](ead::NodeptrT<PybindT>& node)
				{
					return node->get_tensor();
				});
			sess->track(troots);
		},
		"Track node")
		.def("update",
		[](py::object self, std::vector<ead::NodeptrT<PybindT>> nodes)
		{
			auto sess = self.cast<dbg::InteractiveSession*>();
			std::unordered_set<ade::iTensor*> updates;
			for (ead::NodeptrT<PybindT>& node : nodes)
			{
				updates.emplace(node->get_tensor().get());
			}
			sess->update(updates);
		},
		"Return calculated data",
		py::arg("nodes") = std::vector<ead::NodeptrT<PybindT>>{})
		.def("update_target",
		[](py::object self, std::vector<ead::NodeptrT<PybindT>> targeted,
			std::vector<ead::NodeptrT<PybindT>> updated)
		{
			auto sess = self.cast<dbg::InteractiveSession*>();
			std::unordered_set<ade::iTensor*> targets;
			std::unordered_set<ade::iTensor*> updates;
			for (ead::NodeptrT<PybindT>& node : targeted)
			{
				targets.emplace(node->get_tensor().get());
			}
			for (ead::NodeptrT<PybindT>& node : updated)
			{
				updates.emplace(node->get_tensor().get());
			}
			sess->update_target(targets, updates);
		},
		"Calculate node relevant to targets in the graph given list of updated data",
		py::arg("targets"), py::arg("updated") = std::vector<ead::NodeptrT<PybindT>>{})
		.def("join",
		[](py::object self)
		{
			self.cast<dbg::InteractiveSession*>()->join();
		},
		"Wait until session finishes sends all requests")
		.def("stop",
		[](py::object self)
		{
			self.cast<dbg::InteractiveSession*>()->stop();
		},
		"Inform session requests to stop their tasks "
		"(requests will attempt to wrap up call before terminating)")
		.def("optimize",
		[](py::object self, std::string filename)
		{
			auto sess = self.cast<dbg::InteractiveSession*>();
			opt::OptCtx rules = ead::parse_file<PybindT>("cfg/optimizations.rules");
			sess->optimize(rules);
		},
		"Optimize using rules for specified filename");
}
