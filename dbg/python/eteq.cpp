#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

#include "eteq/generated/pyapi.hpp"

#include "dbg/eteq/custom_functor.hpp"

namespace py = pybind11;

using RawDataT = std::vector<PybindT>;
using RawShapeT = std::vector<py::ssize_t>;
using CustomUnaryF = std::function<RawDataT(const RawDataT&,const RawShapeT&)>;

PYBIND11_MODULE(eteq_mocker, m)
{
	m.doc() = "mock eteq equation graphs";

	m.def("custom_unary",
		[](CustomUnaryF unary, NodeptrT& arg)
		{
			return dbg::make_functor<PybindT>(
				[unary](eigen::TensorT<PybindT>& out, const eteq::ArgsT<PybindT>& args)
				{
					const eteq::FuncArg<PybindT>& arg = args.at(0);
					PybindT* data = arg.get_node()->data();
					teq::Shape shape = arg.argshape();
					auto output = unary(
						RawDataT(data, data + shape.n_elems()),
						RawShapeT(shape.begin(), shape.end()));
					auto& slist = out.dimensions();
					teq::Shape outshape(std::vector<teq::DimT>(
						slist.begin(), slist.end()));
					PybindT* outptr = output.data();
					std::copy(outptr, outptr + outshape.n_elems(), out.data());
				},
				{eteq::FuncArg<PybindT>(arg)});
		});
}
