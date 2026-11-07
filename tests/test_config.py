from nvs.config import load_config


def test_grid_is_sorted_and_nonempty():
    g = load_config()["grid"]
    assert g["k"] == sorted(g["k"]) and g["T"] == sorted(g["T"]) and len(g["k"]) > 1
