#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "teq/teq.hpp"

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

// todo: move these to a common file
teq::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return teq::Shape(std::vector<teq::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<teq::DimT> c2pshape (const teq::Shape& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<teq::DimT> fwd(it, et);
	return std::vector<teq::DimT>(fwd.rbegin(), fwd.rend());
}

template <typename T>
py::array typedata_to_array (eteq::iNode<PybindT>* tnode, py::dtype dtype)
{
	auto pshape = c2pshape(tnode->shape());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tnode->data());
}

std::vector<PybindT> arr2vec (teq::Shape& outshape, py::array data)
{
	py::buffer_info info = data.request();
	outshape = p2cshape(info.shape);
	size_t n = outshape.n_elems();
	auto dtype = data.dtype();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	std::vector<PybindT> vec;
	switch (kind)
	{
		case 'f':
			switch (tbytes)
			{
				case 4: // float32
				{
					float* dptr = static_cast<float*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
				break;
				case 8: // float64
				{
					double* dptr = static_cast<double*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported float type with %d bytes", tbytes);
			}
			break;
		case 'i':
			switch (tbytes)
			{
				case 1: // int8
				{
					int8_t* dptr = static_cast<int8_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 2: // int16
				{
					int16_t* dptr = static_cast<int16_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 4: // int32
				{
					int32_t* dptr = static_cast<int32_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				case 8: // int64
				{
					int64_t* dptr = static_cast<int64_t*>(info.ptr);
					vec = std::vector<PybindT>(dptr, dptr + n);
				}
					break;
				default:
					logs::fatalf("unsupported integer type with %d bytes", tbytes);
			}
			break;
		default:
			logs::fatalf("unknown dtype %c", kind);
	}
	return vec;
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
				auto pshape = pyead::c2pshape(shape);
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
		.def("assign",
			[](py::object self, py::array data)
			{
				auto var = self.cast<eteq::VariableNode<PybindT>*>();
				eteq::ShapedArr<PybindT> arr;
				arr.data_ = pyead::arr2vec(arr.shape_, data);
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
					pyead::p2cshape(slist));
			},
			"Return scalar constant node")
		.def("constant",
			[](py::array data)
			{
				teq::Shape shape;
				std::vector<PybindT> vec = pyead::arr2vec(shape, data);
				return eteq::make_constant(vec.data(), shape);
			}, "Return constant node with data")

		// variable creation
		.def("scalar_variable",
			[](PybindT scalar, std::vector<py::ssize_t> slist, std::string label)
			{
				return eteq::make_variable_scalar<PybindT>(scalar, pyead::p2cshape(slist), label);
			},
			"Return labelled variable containing numpy data array",
			py::arg("scalar"),
			py::arg("slist"),
			py::arg("label") = "")
		.def("variable",
			[](py::array data, std::string label)
			{
				teq::Shape shape;
				std::vector<PybindT> vec = pyead::arr2vec(shape, data);
				return eteq::make_variable(vec.data(), shape, label);
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
