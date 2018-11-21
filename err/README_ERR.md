# Error logger (ERR)

Define string manipulation and error handling functionality

ERR provides a global iLogger to users for handling warning, error, and fatal messages

By default, the global logger throws on fatal and prints to stderr for warning and errors

# Extension

Users can implement iLogger and set it as the global logger via set_logger
