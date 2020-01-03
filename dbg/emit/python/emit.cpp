#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "eteq/generated/pyapi.hpp"

#include "dbg/emit/session.hpp"

namespace py = pybind11;

PYBIND11_MODULE(grpc_dbg, m)
{
	m.doc() = "dbg teq equation graphs using interactive grpc session";

	auto isess = (py::class_<teq::iSession>)
		py::module::import("eteq.eteq").attr("iSession");

	py::class_<dbg::InteractiveSession,
		std::shared_ptr<dbg::InteractiveSession>> session(
			m, "InteractiveSession", isess);
	py::implicitly_convertible<teq::iSession,dbg::InteractiveSession>();

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
			[](dbg::InteractiveSession* self, teq::TensptrsT roots)
			{
				self->track(roots);
			},
			"Track node")
		.def("update",
			[](dbg::InteractiveSession* self, std::vector<Tensor> ignored)
			{
				teq::TensSetT ignored_set;
				for (Tensor& node : ignored)
				{
					ignored_set.emplace(node.get());
				}
				self->update(ignored_set);
			},
			"Return calculated data",
			py::arg("ignored") = std::vector<Tensor>{})
		.def("update_target",
			[](dbg::InteractiveSession* self, std::vector<Tensor> targeted,
				std::vector<Tensor> ignored)
			{
				teq::TensSetT targeted_set;
				teq::TensSetT ignored_set;
				for (Tensor& node : targeted)
				{
					targeted_set.emplace(node.get());
				}
				for (Tensor& node : ignored)
				{
					ignored_set.emplace(node.get());
				}
				self->update_target(targeted_set, ignored_set);
			},
			"Calculate node relevant to targets in the graph given list of updated data",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<Tensor>{})
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
			"(requests will attempt to wrap up call before terminating)");
}
