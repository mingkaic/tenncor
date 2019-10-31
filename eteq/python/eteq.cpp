#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "pyutils/convert.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/generated/pyapi.hpp"
#include "eteq/grader.hpp"
#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/functor.hpp"
#include "eteq/session.hpp"
#include "eteq/random.hpp"
#include "eteq/parse.hpp"

namespace py = pybind11;

static const std::unordered_map<std::string,logs::LOG_LEVEL> name2loglevel = {
	{"FATAL", logs::FATAL},
	{"ERROR", logs::ERROR},
	{"WARN", logs::WARN},
	{"INFO", logs::INFO},
	{"DEBUG", logs::DEBUG},
	{"TRACE", logs::TRACE},
};

namespace pyead
{

template <typename T>
py::array typedata_to_array (eteq::iNode<PybindT>* tnode, py::dtype dtype)
{
	auto pshape = pyutils::c2pshape(tnode->shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tnode->data());
}

}

PYBIND11_MODULE(eteq, m)
{
	m.doc() = "eteq variables";

	// ==== data and shape ====
	py::class_<teq::Shape> shape(m, "Shape");
	py::class_<eteq::ShapedArr<PybindT>> sarr(m, "ShapedArr");

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
				eteq::ShapedArr<PybindT> a;
				pyutils::arr2shapedarr(a, ar);
				return a;
			}))
		.def("as_numpy",
			[](eteq::ShapedArr<PybindT>* self) -> py::array
			{
				return pyutils::shapedarr2arr<PybindT>(*self);
			});

	// ==== node ====
	auto node = (py::class_<eteq::iNode<PybindT>,NodeptrT>)
		py::module::import("eteq.tenncor").attr("NodeptrT<PybindT>");

	node
		.def("__str__",
			[](eteq::iNode<PybindT>* self)
			{ return self->to_string(); },
			"Return string representation of internal tensor")
		.def("as_tens",
			[](eteq::iNode<PybindT>* self)
			{ return self->get_tensor(); },
			"Return internal tensor of this node instance")
		.def("shape",
			[](eteq::iNode<PybindT>* self)
			{
				teq::Shape shape = self->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("children",
			[](eteq::iNode<PybindT>* self)
			{
				std::vector<teq::TensptrT> tens;
				if (auto f = dynamic_cast<teq::iFunctor*>(
					self->get_tensor().get()))
				{
					auto args = f->get_children();
					std::transform(args.begin(), args.end(),
					std::back_inserter(tens),
					[](teq::FuncArg& mten)
					{
						return mten.get_tensor();
					});
				}
				return tens;
			})
		.def("get",
			[](eteq::iNode<PybindT>* self)
			{
				return pyead::typedata_to_array<PybindT>(
					self, py::dtype::of<PybindT>());
			});

	// ==== session ====
	py::class_<eteq::iSession> isess(m, "iSession");
	py::class_<eteq::Session> session(m, "Session", isess);

	isess
		.def("track",
			[](eteq::iSession* self, eteq::NodesT<PybindT> roots)
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
			})
		.def("update",
			[](eteq::iSession* self,
				std::vector<NodeptrT> ignored)
			{
				teq::TensSetT ignored_set;
				for (NodeptrT& node : ignored)
				{
					ignored_set.emplace(node->get_tensor().get());
				}
				self->update(ignored_set);
			},
			"Calculate every node in the graph given list of nodes to ignore",
			py::arg("ignored") = std::vector<NodeptrT>{})
		.def("update_target",
			[](eteq::iSession* self,
				std::vector<NodeptrT> targeted,
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
			"Calculate node relevant to targets in the graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<NodeptrT>{});

	py::implicitly_convertible<eteq::iSession,eteq::Session>();
	session
		.def(py::init())
		.def("optimize",
			[](eteq::Session* self, std::string filename)
			{
				opt::OptCtx rules = eteq::parse_file<PybindT>(filename);
				self->optimize(rules);
			},
			py::arg("filename") = "cfg/optimizations.rules",
			"Optimize using rules for specified filename");

	// ==== constant ====
	py::class_<eteq::ConstantNode<PybindT>,std::shared_ptr<eteq::ConstantNode<PybindT>>,
		eteq::iNode<PybindT>> constant(m, "Constant");

	// ==== variable ====
	py::class_<eteq::VariableNode<PybindT>,eteq::VarptrT<PybindT>,
		eteq::iNode<PybindT>> variable(m, "Variable");

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
			[](eteq::VariableNode<PybindT>* self, py::array data)
			{
				eteq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				self->assign(arr);
			},
			"Assign numpy data array to variable");

	// ==== inline functions ====
	m
		// constant creation
		.def("scalar_constant",
			[](PybindT scalar, std::vector<py::ssize_t> slist)
			{
				return eteq::make_constant_scalar<PybindT>(scalar,
					pyutils::p2cshape(slist));
			},
			"Return scalar constant node")
		.def("constant",
			[](py::array data)
			{
				eteq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				return eteq::make_constant(arr.data_.data(), arr.shape_);
			}, "Return constant node with data")

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
				eteq::ShapedArr<PybindT> arr;
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
				eteq::get_engine().seed(seed);
			},
			"Seed internal RNG");
}
