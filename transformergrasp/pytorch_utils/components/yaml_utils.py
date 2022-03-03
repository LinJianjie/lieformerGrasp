import pprint

import yaml


class YAMLConfig:
    def __init__(self, config):
        with open(config) as f:
            self._config = yaml.load(f, Loader=yaml.FullLoader)

    @property
    def config(self):
        return self._config

    def print(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.config)

    def check_key(self, key):
        if key in self.config.keys():
            return True
        else:
            return False

    def get_value(self, key):
        assert self.check_key(key)
        return self.config[key]

    def is_dict(self, check_item):
        return isinstance(check_item, dict)

    def is_list(self, check_item):
        return isinstance(check_item, list)

    def get_all_keys(self):
        for element in self.config.values():
            if self.is_dict(element):
                for k, v in element.items():
                    print(k, ' :', v)
            if self.is_list(element):
                for i in range(len(element)):
                    for k, v in element[i].items():
                        print(k, ' :', v)
