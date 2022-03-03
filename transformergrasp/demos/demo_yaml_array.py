import yaml

if __name__ == '__main__':
    yaml_file = "demo.yaml"
    with open(yaml_file) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    print(len(parameters["SA_modules"]))
