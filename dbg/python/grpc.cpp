#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "eteq/generated/pyapi.hpp"
#include "eteq/eteq.hpp"
#include "eteq/parse.hpp"

#include "dbg/grpc/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(grpc_dbg, m)
{
	m.doc() = "dbg teq equation graphs using interactive grpc session";

	auto isess = (py::class_<eteq::iSession>)
		py::module::import("eteq.eteq").attr("iSession");

	py::class_<dbg::InteractiveSession,
		std::shared_ptr<dbg::InteractiveSession>> session(
			m, "InteractiveSession", isess);
	py::implicitly_convertible<eteq::iSession,dbg::InteractiveSession>();

	m.def("get_isess",
			[](std::string host, size_t request_duration, size_t stream_duration)
			{
				return std::make_shared<dbg::InteractiveSession>(host,
					dbg::ClientConfig{
						std::chrono::milliseconds(request_duration),
						std::chrono::milliseconds(stream_duration),
					});
			},
			py::arg("host") = "localhost:50051",
			py::arg("request_dur") = 1000,
			py::arg("stream_dur") = 30000);
	session
		.def("track",
			[](dbg::InteractiveSession* self, eteq::NodesT<PybindT> roots)
			{
				teq::TensptrsT troots;
				troots.reserve(roots.size());
				std::transform(roots.begin(), roots.end(),
					std::back_inserter(troots),
					[](NodeptrT& node)
					{
						return node->get_tensor();
					});
				self->track(troots);
			},
			"Track node")
		.def("update",
			[](dbg::InteractiveSession* self, std::vector<NodeptrT> ignored)
			{
				teq::TensSetT ignored_set;
				for (NodeptrT& node : ignored)
				{
					ignored_set.emplace(node->get_tensor().get());
				}
				self->update(ignored_set);
			},
			"Return calculated data",
			py::arg("ignored") = std::vector<NodeptrT>{})
		.def("update_target",
			[](dbg::InteractiveSession* self, std::vector<NodeptrT> targeted,
				std::vector<NodeptrT> ignored)
			{
				teq::TensSetT targeted_set;
				teq::TensSetT ignored_set;
				for (NodeptrT& node : targeted)
				{
					targeted_set.emplace(node->get_tensor().get());
				}
				for (NodeptrT& node : ignored)
				{
					ignored_set.emplace(node->get_tensor().get());
				}
				self->update_target(targeted_set, ignored_set);
			},
			"Calculate node relevant to targets in the graph given list of updated data",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<NodeptrT>{})
		.def("join",
			[](dbg::InteractiveSession* self)
			{
				self->join();
			},
			"Wait until session finishes sends all requests")
		.def("stop",
			[](dbg::InteractiveSession* self)
			{
				self->stop();
			},
			"Inform session requests to stop their tasks "
			"(requests will attempt to wrap up call before terminating)")
		.def("optimize",
			[](dbg::InteractiveSession* self, std::string filename)
			{
				opt::OptCtx rules = eteq::parse_file<PybindT>("cfg/optimizations.rules");
				self->optimize(rules);
			},
			"Optimize using rules for specified filename");
}
