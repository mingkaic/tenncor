#include "pybind11/stl.h"

#include "pyutils/convert.hpp"

#include "teq/session.hpp"

#include "eteq/generated/pyapi.hpp"
#include "eteq/grader.hpp"
#include "eteq/make.hpp"
#include "eteq/placeholder.hpp"
#include "eteq/optimize.hpp"

namespace py = pybind11;

namespace pyead
{

template <typename T>
py::array typedata_to_array (eteq::iLink<PybindT>* link, py::dtype dtype)
{
	auto pshape = pyutils::c2pshape(link->shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		link->data());
}

}

PYBIND11_MODULE(eteq, m)
{
	m.doc() = "eteq variables";

	// ==== data and shape ====
	py::class_<teq::Shape> shape(m, "Shape");
	py::class_<teq::ShapedArr<PybindT>> sarr(m, "ShapedArr");

	shape
		.def(py::init(
			[](std::vector<py::ssize_t> dims)
			{
				return pyutils::p2cshape(dims);
			}))
		.def("__getitem__",
			[](teq::Shape& shape, size_t idx) { return shape.at(idx); },
			py::is_operator())
		.def("n_elems",
			[](teq::Shape& shape) { return shape.n_elems(); })
		.def("as_list",
			[](teq::Shape& shape) { return pyutils::c2pshape(shape); });

	sarr
		.def(py::init(
			[](py::array ar)
			{
				teq::ShapedArr<PybindT> a;
				pyutils::arr2shapedarr(a, ar);
				return a;
			}))
		.def("as_numpy",
			[](teq::ShapedArr<PybindT>* self) -> py::array
			{
				return pyutils::shapedarr2arr<PybindT>(*self);
			});

	// ==== link ====
	auto link = (py::class_<eteq::iLink<PybindT>,LinkptrT>)
		py::module::import("eteq.tenncor").attr("LinkptrT<PybindT>");

	link
		.def("as_tens",
			[](eteq::iLink<PybindT>* self)
			{ return self->get_tensor(); },
			"Return internal tensor of this link instance")
		.def("shape",
			[](eteq::iLink<PybindT>* self)
			{
				teq::Shape shape = self->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("get",
			[](eteq::iLink<PybindT>* self)
			{
				return pyead::typedata_to_array<PybindT>(
					self, py::dtype::of<PybindT>());
			});

	// ==== session ====
	py::class_<teq::iSession> isess(m, "iSession");
	py::class_<teq::Session> session(m, "Session", isess);

	isess
		.def("track",
			[](teq::iSession* self, eteq::LinksT<PybindT> roots)
			{
				teq::TensptrsT troots;
				troots.reserve(roots.size());
				std::transform(roots.begin(), roots.end(),
					std::back_inserter(troots),
					[](LinkptrT& link)
					{
						return link->get_tensor();
					});
				self->track(troots);
			})
		.def("update",
			[](teq::iSession* self,
				std::vector<LinkptrT> ignored)
			{
				teq::TensSetT ignored_set;
				for (LinkptrT& link : ignored)
				{
					ignored_set.emplace(link->get_tensor().get());
				}
				self->update(ignored_set);
			},
			"Calculate every link in the graph given list of nodes to ignore",
			py::arg("ignored") = std::vector<LinkptrT>{})
		.def("update_target",
			[](teq::iSession* self,
				std::vector<LinkptrT> targeted,
				std::vector<LinkptrT> ignored)
			{
				teq::TensSetT targeted_set;
				teq::TensSetT ignored_set;
				for (LinkptrT& link : targeted)
				{
					targeted_set.emplace(link->get_tensor().get());
				}
				for (LinkptrT& link : ignored)
				{
					ignored_set.emplace(link->get_tensor().get());
				}
				self->update_target(targeted_set, ignored_set);
			},
			"Calculate link relevant to targets in the graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<LinkptrT>{})
		.def("get_tracked", &teq::iSession::get_tracked);

	py::implicitly_convertible<teq::iSession,teq::Session>();
	session
		.def(py::init());

	// ==== variable ====
	py::class_<eteq::Variable<PybindT>,eteq::VarptrT<PybindT>,
		teq::iTensor> variable(m, "Variable");

	variable
		.def(py::init(
			[](std::vector<py::ssize_t> slist, PybindT scalar, std::string label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyutils::p2cshape(slist), label);
			}),
			py::arg("shape"),
			py::arg("scalar") = 0,
			py::arg("label") = "")
		.def("assign",
			[](eteq::Variable<PybindT>* self, py::array data)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				self->assign(arr);
			},
			"Assign numpy data array to variable");

	// ==== placeholder ====
	py::class_<eteq::PlaceLink<PybindT>,eteq::PlaceLinkptrT<PybindT>,
		eteq::iLink<PybindT>> placeholder(m, "PlaceLink");

	placeholder
		.def(py::init(
			[](std::vector<py::ssize_t> slist, std::string label)
			{
				return std::make_shared<eteq::PlaceLink<PybindT>>(
					std::make_shared<teq::Placeholder>(
						pyutils::p2cshapesign(slist), label));
			}),
			py::arg("shape"),
			py::arg("label") = "")
		.def("assign",
			[](eteq::PlaceLink<PybindT>* self, py::array data)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				self->assign(arr);
			},
			"Assign numpy data array to placeholder")
		.def("assign",
			[](eteq::PlaceLink<PybindT>* self, LinkptrT link)
			{
				self->assign(link);
			},
			"Assign numpy data array to placeholder");

	// ==== inline functions ====
	m
		.def("link",
			[](teq::TensptrT tens)
			{
				return eteq::to_link<PybindT>(tens);
			})

		// constant creation
		.def("scalar_constant",
			[](PybindT scalar, std::vector<py::ssize_t> slist)
			{
				return eteq::make_constant_scalar<PybindT>(scalar,
					pyutils::p2cshape(slist));
			},
			"Return scalar constant link")
		.def("constant",
			[](py::array data)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				return eteq::make_constant(arr.data_.data(), arr.shape_);
			}, "Return constant link with data")

		// variable creation
		.def("scalar_variable",
			[](PybindT scalar, std::vector<py::ssize_t> slist, std::string label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyutils::p2cshape(slist), label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("slist"),
			py::arg("label") = "")
		.def("variable",
			[](py::array data, std::string label)
			{
				teq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				return eteq::make_variable(arr.data_.data(), arr.shape_, label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("data"),
			py::arg("label") = "")

		// other stuff
		.def("derive", &eteq::derive<PybindT>,
			"Return derivative of first tensor with respect to second tensor (deprecated)")
		.def("seed",
			[](size_t seed)
			{
				eigen::get_engine().seed(seed);
			},
			"Seed internal RNG");

	// optimization rules
	py::class_<opt::CversionCtx> rules(m, "OptRules");

	m
		.def("parse_optrules", &eteq::parse_file<PybindT>,
			py::arg("filename") = "cfg/optimizations.rules",
			"Optimize using rules for specified filename")
		.def("optimize",
			[](teq::iSession& sess, opt::CversionCtx rules)
			{
				eteq::optimize<PybindT>(sess, rules);
			});
}
