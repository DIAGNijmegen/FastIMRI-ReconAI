import importlib.resources

config_debug = importlib.resources.read_text(__package__, 'config_debug.yaml')
config_default = importlib.resources.read_text(__package__, 'config_default.yaml')
