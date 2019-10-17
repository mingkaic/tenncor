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

	// ==== node ====
	auto node = (py::class_<eteq::iNode<PybindT>,NodeptrT>)
		py::module::import("eteq.tenncor").attr("NodeptrT<PybindT>");

	node
		.def("__str__",
			[](py::object self)
			{ return self.cast<eteq::iNode<PybindT>*>()->to_string(); },
			"Return string representation of internal tensor")
		.def("as_tens",
			[](py::object self)
			{ return self.cast<eteq::iNode<PybindT>*>()->get_tensor(); },
			"Return internal tensor of this node instance")
		.def("shape",
			[](py::object self)
			{
				teq::Shape shape = self.cast<eteq::iNode<PybindT>*>()->shape();
				auto pshape = pyutils::c2pshape(shape);
				std::vector<int> ipshape(pshape.begin(), pshape.end());
				return py::array(ipshape.size(), ipshape.data());
			},
			"Return this instance's shape")
		.def("children",
			[](py::object self)
			{
				std::vector<teq::TensptrT> tens;
				if (auto f = dynamic_cast<teq::iFunctor*>(
					self.cast<eteq::iNode<PybindT>*>()->get_tensor().get()))
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
			[](py::object self)
			{
				return pyead::typedata_to_array<PybindT>(
					self.cast<eteq::iNode<PybindT>*>(),
					py::dtype::of<PybindT>());
			});

	// ==== session ====
	py::class_<eteq::iSession> isess(m, "iSession");
	py::class_<eteq::Session> session(m, "Session", isess);

	isess
		.def("track",
			[](py::object self, eteq::NodesT<PybindT> roots)
			{
				auto sess = self.cast<eteq::iSession*>();
				teq::TensptrsT troots;
				troots.reserve(roots.size());
				std::transform(roots.begin(), roots.end(),
					std::back_inserter(troots),
					[](NodeptrT& node)
					{
						return node->get_tensor();
					});
				sess->track(troots);
			})
		.def("update",
			[](py::object self,
				std::vector<NodeptrT> ignored)
			{
				auto sess = self.cast<eteq::iSession*>();
				teq::TensSetT ignored_set;
				for (NodeptrT& node : ignored)
				{
					ignored_set.emplace(node->get_tensor().get());
				}
				sess->update(ignored_set);
			},
			"Calculate every node in the graph given list of nodes to ignore",
			py::arg("ignored") = std::vector<NodeptrT>{})
		.def("update_target",
			[](py::object self,
				std::vector<NodeptrT> targeted,
				std::vector<NodeptrT> ignored)
			{
				auto sess = self.cast<eteq::iSession*>();
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
				sess->update_target(targeted_set, ignored_set);
			},
			"Calculate node relevant to targets in the graph given list of nodes to ignore",
			py::arg("targeted"),
			py::arg("ignored") = std::vector<NodeptrT>{});

	py::implicitly_convertible<eteq::iSession,eteq::Session>();
	session
		.def(py::init())
		.def("optimize",
			[](py::object self, std::string filename)
			{
				auto sess = self.cast<eteq::Session*>();
				opt::OptCtx rules = eteq::parse_file<PybindT>(filename);
				sess->optimize(rules);
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
			[](py::object self, py::array data)
			{
				auto var = self.cast<eteq::VariableNode<PybindT>*>();
				eteq::ShapedArr<PybindT> arr;
				pyutils::arr2shapedarr(arr, data);
				var->assign(arr);
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
