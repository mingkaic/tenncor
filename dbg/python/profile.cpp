#include <fstream>
#include <sstream>

#include <curl/curl.h>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "dbg/profile/profile.hpp"

#include "tenncor/tenncor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tenncor_profile, m)
{
	m.doc() = "profile teq graphs";

	m
		// ==== to stdout functions ====
		.def("profile",
		[](const std::string& filename, eteq::ETensorsT roots)
		{
			teq::TensT rs =
				teq::multi_get<teq::iTensor,eteq::ETensor>(roots);
			dbg::profile::gexf_write(filename, rs);
		},
		"Profile graph of tensors and print to file",
		py::arg("filename"),
		py::arg("roots"))
		// ==== remote functions ====
		.def("remote_profile",
		[](const std::string& addr, eteq::ETensorsT roots,
			const std::string& outdir)
		{
			CURL* curl = curl_easy_init();
			if (curl)
			{
				teq::TensT rs =
					teq::multi_get<teq::iTensor,eteq::ETensor>(roots);

				std::stringstream ss;
				dbg::profile::gexf_write(ss, rs, outdir);

				std::string target = addr + "/graph";

				std::string buf = ss.str();
				curl_easy_setopt(curl, CURLOPT_URL, target.c_str());
				curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buf.size());
				curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buf.c_str());

				curl_easy_perform(curl);
			}
		},
		"Profile graph of tensors",
		py::arg("addr"),
		py::arg("roots"),
		py::arg("outdir") = "/tmp")
		.def("remote_profile2", dbg::profile::remote_profile,
		"Profile graph of tensors and report to remote address",
		py::arg("addr"),
		py::arg("roots"));
}
