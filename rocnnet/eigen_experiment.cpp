#include "ead/ead.hpp"
#include "dbg/tensor.hpp"

int main (int argc, char** argv)
{
	// in, width, height, batch
    ade::Shape shape({2, 3, 3, 3});
	std::vector<float> data(shape.n_elems());
	std::iota(data.begin(), data.end(), 1);

	// out, in, width, height
    ade::Shape kshape({4, 2, 2, 2});
	std::vector<float> kdata(kshape.n_elems());
	std::iota(kdata.begin(), kdata.end(), 1);

	{
		ead::Session<float> session;
		ead::NodeptrT<float> image = ead::make_constant<float>(data.data(), shape);
		ead::NodeptrT<float> kernel = ead::make_constant<float>(kdata.data(), kshape);

		ade::DimT nfilters = kernel->shape().at(0);
		ead::NodesT<float> paddeds;
		paddeds.reserve(nfilters);
		for (ade::DimT i = 0; i < nfilters; ++i)
		{
			auto filter = age::slice(kernel, i, 1, 0);
			filter = age::permute(filter, {1, 2, 3, 0});
			auto conved = age::convolution(image, filter, {0, 1, 2});
			auto padded = age::pad(conved, {i, nfilters - i - 1}, 0);
			paddeds.push_back(padded);
		}

		ead::NodeptrT<float> out = paddeds[0];
		for (ade::DimT i = 1; i < nfilters; ++i)
		{
			out = age::add(out, paddeds[i]);
		}

		session.track(out);
		session.update();
		float* conved_data = (float*) out->data();
		auto conved_shape = out->shape();
		std::cout << conved_shape.to_string() << "\n";

		std::vector<uint8_t> slist(conved_shape.begin(), conved_shape.end());
		PrettyTensor<float> printer({6, 6, 6, 6});
		printer.print(std::cout, conved_data, slist);
		std::cout << "\n";
	}

    return 0;
}
