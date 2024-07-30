from sconf import Config

config_file = "./config.yaml"

config = Config(config_file)

print(config.model)