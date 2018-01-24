//
// Created by Mingkai Chen on 2017-01-30.
//

#include "include/edgeinfo/csv_record/csv_record.hpp"

#if defined(CSV_RCD) && defined(CSV_RECORD_HPP)

namespace rocnnet_record
{

std::unique_ptr<igraph_record> record_status::rec =
	std::make_unique<csv_record>("op-profile.csv");
bool record_status::rec_good = true;

csv_record::csv_record (std::string fname) :
	outname_(fname) {}

void csv_record::setVerbose (bool verbosity)
{
	verbose_ = verbosity;
}

void csv_record::setDisplayShape (bool display)
{
	display_shape_ = display;
}

void csv_record::to_csv (const nnet::iconnector* consider_graph) const
{
	std::ofstream ofile;
	ofile.open(outname_);
	size_t num_des = 0;
	std::unordered_map<const nnet::inode*, size_t> num_corres;
	if (ofile.is_open())
	{
		for (auto sub2obs : this->subj_nodes)
		{
			const nnet::subject* sbs = sub2obs.first;
			const nnet::inode* sub = dynamic_cast<const nnet::inode*>(sbs);
			if (nullptr == sub)
			{
				continue; // skip if not inode of type T
			}
			auto& obs_infos = sub2obs.second;
			for (auto info : obs_infos)
			{
				const nnet::iconnector* obs = static_cast<const nnet::iconnector*>(info.obs_);
				if (consider_graph && !obs->is_same_graph(consider_graph))
				{
					continue;
				}

				std::stringstream obstrm;
				std::stringstream sbstrm;

				if (verbose_)
				{
					obstrm << obs->get_name();
					sbstrm << sub->get_name();
				}
				else
				{
					auto oit = num_corres.find(obs);
					auto sit = num_corres.find(sub);

					size_t obidx;
					if (num_corres.end() == oit)
					{
						num_corres[obs] = obidx = num_des++;
					}
					else
					{
						obidx = oit->second;
					}
					size_t sbidx;
					if (num_corres.end() == sit)
					{
						num_corres[sub] = sbidx = num_des++;
					}
					else
					{
						sbidx = sit->second;
					}

					obstrm << '[' << obidx << ']' << obs->get_label();
					sbstrm << '[' << sbidx << ']' << sub->get_label();
				}
				if (display_shape_)
				{
					obstrm << "(";
					sbstrm << "(";
					print_shape(obs->get_shape(), obstrm);
					print_shape(sub->get_shape(), sbstrm);
					obstrm << ")";
					sbstrm << ")";
				}

				ofile << obstrm.str() << "," << sbstrm.str() << "," << info.idx_ << "\n";
			}
		}
	}
	ofile.close();
}

}

#endif