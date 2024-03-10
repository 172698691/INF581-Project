class NamedDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f'No key: {name} in NamedDict. ')

    def __setattr__(self, key, value):
        self[key] = value


environ_config = NamedDict({
    'reward_params': (2, 8, 25, 0.1, -0.6),
})
