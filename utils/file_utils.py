import pickle


def save_pkl(filename, obj) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def load_pkl(filename):
    with open(filename, "rb") as handle:
        return pickle.load(handle)
