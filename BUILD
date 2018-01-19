cc_library(
	name = "tenncor",
	hdrs = glob([ "include/**/*.hpp" ]) + glob([ "src/**/*.ipp" ]),
	srcs = glob([ "src/**/*.cpp" ]),
	deps = [ "//proto:tenncor_serial_cc_proto" ],
	linkstatic = 1,
	copts = [ "-std=c++14" ],
	visibility = [ "//visibility:public" ],
)
