#include "rocnnet/layr/ilayer.hpp"

struct ConvLayer : public iLayer
{
	ConvLayer (std::pair<size_t,size_t> filter_hw,
		size_t in_ncol, size_t out_ncol, std::string label) :
		iLayer(label)
	{
		ade::Shape shape shape({filter_hw.first, filter_hw.second, in_ncol, out_ncol});
		size_t ndata = shape.n_elems();

		size_t input_size = filter_hw.first * filter_hw.second * in_ncol;
		double bound = 1.0 / std::sqrt(input_size);
		std::uniform_real_distribution<double> dist(-bound, bound);
		auto gen = [&dist]()
		{
			return dist(util::get_engine());
		};
		std::vector<double> data(ndata);
		std::generate(data.begin(), data.end(), gen);

		weight_ = llo::Source<double>::get(shape, data);
		bias_ = llo::Source<double>::get(ade::Shape({out_ncol}),
			std::vector<double>(out_ncol, 0));
	}

	virtual ~ConvLayer (void) {}

	llo::DataNode operator () (llo::DataNode input)
	{
		return add(convolute(input, weight_), bias_);
	}

	std::vector<llo::DataNode> get_variables (void) const override
	{
		return {weight_, bias_};
	}

protected:
	llo::DataNode weight_;

	llo::DataNode bias_;
};
