workspace(name = "com_github_mingkaic_tenncor")

# protobuf dependency
http_archive(
	name = "com_google_protobuf",
	sha256 = "cef7f1b5a7c5fba672bec2a319246e8feba471f04dcebfe362d55930ee7c1c30",
	strip_prefix = "protobuf-3.5.0",
	urls = ["https://github.com/google/protobuf/archive/v3.5.0.zip"],
)

# gtest dependency
new_git_repository(
	name = "googletest",
	build_file = "BUILD.gmock",
	remote = "https://github.com/google/googletest",
	tag = "release-1.8.0",
)
