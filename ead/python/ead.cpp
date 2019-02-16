#include <fstream>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "ade/ade.hpp"

#include "ead/generated/api.hpp"

#include "ead/grader.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/functor.hpp"
#include "ead/session.hpp"

namespace py = pybind11;

namespace pyead
{

ade::Shape p2cshape (std::vector<py::ssize_t>& pyshape)
{
	return ade::Shape(std::vector<ade::DimT>(
		pyshape.rbegin(), pyshape.rend()));
}

std::vector<ade::DimT> c2pshape (const ade::Shape& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<ade::DimT> fwd(it, et);
	return std::vector<ade::DimT>(fwd.rbegin(), fwd.rend());
}

std::vector<ade::DimT> c2pshape (
	const Eigen::DSizes<long,ade::rank_cap>& cshape)
{
	auto it = cshape.begin();
	auto et = cshape.end();
	while (it != et && *(et-1) == 1)
	{
		--et;
	}
	std::vector<ade::DimT> fwd(it, et);
	return std::vector<ade::DimT>(fwd.rbegin(), fwd.rend());
}

template <typename T>
py::array typedata_to_array (ead::TensMapT<T>& tdata, py::dtype dtype)
{
	auto pshape = pyead::c2pshape(tdata.dimensions());
	return py::array(dtype,
		py::array::ShapeContainer(pshape.begin(), pshape.end()),
		tdata.data());
}

std::vector<double> arr2vec (ade::Shape& outshape, py::array data)
{
	py::buffer_info info = data.request();
	outshape = pyead::p2cshape(info.shape);
	size_t n = outshape.n_elems();
	auto dtype = data.dtype();
	char kind = dtype.kind();
	py::ssize_t tbytes = dtype.itemsize();
	assert(kind == 'f');
	std::vector<double> vec;
	switch (tbytes)
	{
		case 4: // float32
		{
			float* dptr = static_cast<float*>(info.ptr);
			vec = std::vector<double>(dptr, dptr + n);
		}
		break;
		case 8: // float64
		{
			double* dptr = static_cast<double*>(info.ptr);
			vec = std::vector<double>(dptr, dptr + n);
		}
		break;
		default:
			logs::fatalf("unsupported float type with %d bytes", tbytes);
	}
	return vec;
}

}

PYBIND11_MODULE(ead, m)
{
	m.doc() = "ead variables";

	// ==== node ====
	py::object node = (py::object)
		py::module::import("ead.age").attr("NodeptrT<double>");

	((py::class_<ead::iNode<double>,ead::NodeptrT<double>>) node)
		.def("__str__",
		[](py::object self)
		{
			auto dnode = self.cast<ead::iNode<double>*>();
			return dnode->get_tensor()->to_string();
		},
		"Return string representation of this tensor instance")
		.def("shape",
		[](py::object self)
		{
			auto dnode = self.cast<ead::iNode<double>*>();
			ade::Shape shape = dnode->get_tensor()->shape();
			auto pshape = pyead::c2pshape(shape);
			std::vector<int> ipshape(pshape.begin(), pshape.end());
			return py::array(ipshape.size(), ipshape.data());
		},
		"Return this instance's shape")
		.def("children",
		[](py::object self)
		{
			auto dnode = self.cast<ead::iNode<double>*>();
			std::vector<ade::TensptrT> tens;
			if (auto f = dynamic_cast<ade::iFunctor*>(
				dnode->get_tensor().get()))
			{
				auto args = f->get_children();
				std::transform(args.begin(), args.end(),
				std::back_inserter(tens),
				[](ade::FuncArg& mten)
				{
					return mten.get_tensor();
				});
			}
			return tens;
		})
		.def("as_tens",
		[](py::object self)
		{
			auto dnode = self.cast<ead::iNode<double>*>();
			return dnode->get_tensor();
		})
		.def("get",
		[](py::object self)
		{
			auto dnode = self.cast<ead::iNode<double>*>();
			auto tmap = dnode->get_tensmap();
			return pyead::typedata_to_array<double>(*tmap, py::dtype::of<double>());
		});

	// ==== session ====
	py::class_<ead::Session<double>> session(m, "Session");
	session
		.def(py::init())
		.def("track", &ead::Session<double>::track, "Track node")
		.def("update",
		[](py::object self, std::vector<ead::NodeptrT<double>> nodes)
		{
			auto sess = self.cast<ead::Session<double>*>();
			std::unordered_set<ade::iTensor*> updates;
			for (ead::NodeptrT<double>& node : nodes)
			{
				updates.emplace(node->get_tensor().get());
			}
			sess->update(updates);
		},
		"Return calculated data",
		py::arg("nodes") = std::vector<ead::NodeptrT<double>>{});

	// ==== constant ====
	py::class_<ead::ConstantNode<double>,std::shared_ptr<ead::ConstantNode<double>>> constant(
		m, "Constant", node);

	py::implicitly_convertible<ead::iNode<double>,ead::ConstantNode<double>>();

	// ==== variable ====
	py::class_<ead::VariableNode<double>,ead::VarptrT<double>> variable(
		m, "Variable", node);

	py::implicitly_convertible<ead::iNode<double>,ead::VariableNode<double>>();

	variable
		.def("assign",
		[](py::object self, py::array data)
		{
			auto var = self.cast<ead::VariableNode<double>*>();
			ade::Shape shape;
			std::vector<double> vec = pyead::arr2vec(shape, data);
			var->assign(vec.data(), shape);
		},
		"Assign numpy data array to variable");

	// ==== inline functions ====
	m.def("scalar_constant", &ead::make_constant_scalar<double>,
	"Return scalar constant node");

	m.def("constant",
	[](py::array data)
	{
		ade::Shape shape;
		std::vector<double> vec = pyead::arr2vec(shape, data);
		return ead::make_constant(vec.data(), shape);
	}, "Return constant node with data");

	m.def("scalar_variable", &ead::make_variable_scalar<double>,
	"Return labelled variable containing numpy data array",
	py::arg("scalar"), py::arg("shape"), py::arg("label") = "");

	m.def("variable",
	[](py::array data, std::string label)
	{
		ade::Shape shape;
		std::vector<double> vec = pyead::arr2vec(shape, data);
		return ead::make_variable(vec.data(), shape, label);
	},
	"Return labelled variable containing numpy data array",
	py::arg("data"), py::arg("label") = "");


	m.def("derive", &ead::derive<double>,
	"Return derivative of first tensor with respect to second tensor");

	m.def("seed",
	[](size_t seed)
	{
		ead::get_engine().seed(seed);
	},
	"Seed internal RNG");
}
