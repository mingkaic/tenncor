cc_library(
	name = "tenncor",
	hdrs = glob(["include/**/*.hpp"]) + glob(["src/**/*.ipp"]),
	srcs = glob(["src/**/*.cpp"]),
	deps = [
		"@boost//:uuid",
		"//proto:tenncor_monitor_cc_proto",
		"//proto:tenncor_serial_cc_proto",
	],
	copts = ["-std=c++14"]
)
