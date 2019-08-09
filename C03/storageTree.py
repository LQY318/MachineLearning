def store_tree(tree, filename):
    """
    Function to store tree sturcture.

    :args tree: dict, tree structure
    :args filename: string, filename of storing tree structure
    """
    import pickle
    with open(filename, "wb+") as file:  # 使用二进制打开，"w"会报错
        pickle.dump(tree, file)

def grab_tree(filename):
    import pickle
    with open(filename, "rb+") as file:  # 使用二进制打开，"r"会报错
        tree = pickle.load(file)
    return tree


if __name__ == "__main__":
    tree = {'no surfacing': {'1': {'flippers': {'1': 'yes', '0': 'no'}}, '0': 'no'}}
    filename = "demo.txt"
    # store_tree(tree, filename)
    print(grab_tree(filename))